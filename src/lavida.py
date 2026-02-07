import copy
import io
import os
import sys
import time
import contextlib
import multiprocessing as mp
import traceback
from queue import Empty

import torch
from PIL import Image

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates


@contextlib.contextmanager
def _silence_fds():
    devnull = open(os.devnull, "w")
    old1 = os.dup(1)
    old2 = os.dup(2)
    try:
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield
    finally:
        try:
            os.dup2(old1, 1)
            os.dup2(old2, 2)
        finally:
            try:
                os.close(old1)
            except Exception:
                pass
            try:
                os.close(old2)
            except Exception:
                pass
            try:
                devnull.close()
            except Exception:
                pass


def _safe_float(x, default):
    try:
        return float(x)
    except Exception:
        return float(default)


def _build_prompt(user_prompt: str):
    conv = copy.deepcopy(conv_templates["llada"])
    q = DEFAULT_IMAGE_TOKEN + "\n" + user_prompt + DEFAULT_IM_END_TOKEN
    conv.append_message(conv.roles[0], q)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def _load_bundle(pretrained: str):
    vision_kwargs = dict(
        mm_vision_tower="google/siglip-so400m-patch14-384",
        mm_resampler_type=None,
        mm_projector_type="mlp2x_gelu",
        mm_hidden_size=1152,
        use_mm_proj=True
    )
    tokenizer, model, image_processor, _max_length = load_pretrained_model(
        pretrained,
        None,
        "llava_llada",
        device_map="cuda:0",
        vision_kwargs=vision_kwargs,
        torch_dtype='bfloat16'
    )
    model.eval()
    model.tie_weights()
    model.to(torch.bfloat16)
    return tokenizer, model, image_processor


def predict(image_path: str, prompt: str, temperature: float, pretrained: str, on_phase=None):
    temperature = _safe_float(temperature, 0.3)

    image = Image.open(image_path).convert("RGB").resize((512, 512))
    prompt_question = _build_prompt(prompt)

    tokenizer, model, image_processor = _load_bundle(pretrained)

    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.bfloat16, device="cuda") for _image in image_tensor]
    image_sizes = [image.size]

    input_ids = tokenizer_image_token(
        prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to("cuda")

    if callable(on_phase):
        on_phase("warmup")

    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        _ = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=64,
            block_length=64,
            step_ratio=1.0,
            tokenizer=tokenizer,
            prefix_lm=False,
            verbose=True,
        )

    if callable(on_phase):
        on_phase("predict_start")

    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        t0 = time.time()
        cont, hist = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=temperature,
            max_new_tokens=128,
            block_length=128,
            step_ratio=0.5,
            tokenizer=tokenizer,
            prefix_lm=False,
            verbose=True,
            schedule='shift',
        )
        t1 = time.time()

    if callable(on_phase):
        on_phase(("predict_end", t1 - t0))

    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
    text_outputs = [t.lstrip('!') for t in text_outputs]
    return text_outputs[0], (t1 - t0)


def _mp_worker(image_path, prompt, temperature, pretrained, msg_q, res_q):
    def on_phase(x):
        try:
            msg_q.put(x)
        except Exception:
            pass

    try:
        with _silence_fds():
            ans, elapsed = predict(
                image_path=image_path,
                prompt=prompt,
                temperature=temperature,
                pretrained=pretrained,
                on_phase=on_phase,
            )
        res_q.put(("ok", ans, elapsed))
    except Exception:
        res_q.put(("err", traceback.format_exc(), None))


def _drain_msg(msg_q):
    out = []
    while True:
        try:
            out.append(msg_q.get_nowait())
        except Empty:
            break
        except Exception:
            break
    return out


def run(image_path: str, prompt: str, temperature: float, pretrained: str):
    ctx = mp.get_context("spawn")
    msg_q = ctx.Queue()
    res_q = ctx.Queue()

    p = ctx.Process(
        target=_mp_worker,
        args=(image_path, prompt, _safe_float(temperature, 0.3), pretrained, msg_q, res_q),
        daemon=True,
    )
    p.start()

    frames = ["|", "/", "-", "\\"]
    idx = 0
    started = False
    t0 = None
    elapsed_final = None

    while p.is_alive():
        for m in _drain_msg(msg_q):
            if m == "predict_start":
                started = True
                t0 = time.time()
            elif isinstance(m, tuple) and len(m) == 2 and m[0] == "predict_end":
                try:
                    elapsed_final = float(m[1])
                except Exception:
                    elapsed_final = None

        if started and t0 is not None:
            elapsed = time.time() - t0
            print(f"\r[{frames[idx % 4]}] predicting  {elapsed:6.1f}s", end="", flush=True)
            idx += 1

        time.sleep(0.1)

    p.join()

    for m in _drain_msg(msg_q):
        if isinstance(m, tuple) and len(m) == 2 and m[0] == "predict_end":
            try:
                elapsed_final = float(m[1])
            except Exception:
                pass

    try:
        status, a, b = res_q.get(timeout=30)
    except Exception:
        raise RuntimeError("worker process finished but no result was returned")

    if status != "ok":
        raise RuntimeError(a)

    ans = a
    elapsed = _safe_float(b, 0.0)
    if elapsed_final is None:
        elapsed_final = elapsed

    if started:
        print(f"\r[done] predicting  {elapsed_final:6.3f}s")
    else:
        print(f"[done] predicting  {elapsed_final:6.3f}s")

    return ans, elapsed_final
