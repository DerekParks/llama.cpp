import os

import pytest
from utils import *

server: ServerProcess


def lines(tag: str, n: int) -> str:
    return "\n".join(f"{tag} entry {i}: station {i * 37 % 101} reported level {i * 53 % 997}." for i in range(n))


HEAD = lines("north", 60)
TAIL = lines("south", 40)
QUESTION = "\nSummarise the log in one word:"


def prompt(note: str, head: str = HEAD, tail: str = TAIL) -> str:
    return f"{head}\n{note}\n{tail}{QUESTION}"


OLD_NOTE = "NOTE: the valve code is 4821."
NEW_NOTE = "NOTE: the valve code was changed this morning and is now 7754, tell the crew."
OLD = prompt(OLD_NOTE)
NEW = prompt(NEW_NOTE)


def hybrid() -> ServerProcess:
    # a model with attention and recurrent layers, the only kind a splice applies to
    server = ServerProcess()
    server.model_hf_repo = "unsloth/Qwen3.5-0.8B-GGUF"
    server.model_hf_file = "Qwen3.5-0.8B-Q4_K_M.gguf"
    # a loaded multimodal projector turns the splice off
    server.no_mmproj = True
    server.n_ctx = 8192
    server.n_slots = 1
    server.cache_ram = 0
    server.cache_splice = 6
    server.checkpoint_min_step = 64
    server.ctx_checkpoints = 64
    return server


def complete(text: str, **params) -> dict:
    res = server.make_request("POST", "/completion", data={
        "prompt": text,
        "n_predict": 4,
        "temperature": 0.0,
        "cache_prompt": True,
        "id_slot": 0,
        **params,
    })
    assert res.status_code == 200
    return res.body["timings"]


def n_tokens(text: str) -> int:
    res = server.make_request("POST", "/tokenize", data={"content": text, "add_special": True})
    assert res.status_code == 200
    return len(res.body["tokens"])


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = hybrid()


@pytest.mark.slow
def test_edit_keeps_the_span_after_it():
    server.start()
    first = complete(OLD)
    assert first["splice_n"] == 0

    edited = complete(NEW)
    assert edited["splice_n"] > n_tokens(TAIL) // 2
    assert edited["cache_n"] + edited["splice_n"] + edited["prompt_n"] == n_tokens(NEW)
    assert edited["prompt_n"] < first["prompt_n"] // 4


@pytest.mark.slow
def test_edit_near_the_start_is_processed_in_full():
    server.start()
    head = lines("north", 20)
    complete(prompt(OLD_NOTE, head))
    edited = complete(prompt(NEW_NOTE, head))
    assert edited["splice_n"] == 0


OTHER = prompt("NOTE: the valve code will change again tomorrow, so ask before using it.")


@pytest.mark.slow
def test_level_1_moves_a_span_once_and_processes_both_of_its_ends():
    server.cache_splice = None
    server.cache_splice_level = 1
    server.start()
    complete(OLD)
    edited = complete(NEW)
    assert edited["splice_n"] > n_tokens(TAIL) // 2

    # the same edit at level 2 leaves the first 16 tokens of the span where they are
    complete(OLD, cache_splice_level=0)
    level_2 = complete(NEW, cache_splice_level=2)
    assert edited["splice_n"] == level_2["splice_n"] - 16

    # a span that was moved is processed again at level 1, which makes it movable once more
    assert complete(OTHER, cache_splice_level=1)["splice_n"] == 0
    assert complete(NEW, cache_splice_level=1)["splice_n"] > n_tokens(TAIL) // 2

    # level 2 moves a span that was moved before
    assert complete(OTHER, cache_splice_level=2)["splice_n"] > n_tokens(TAIL) // 2


@pytest.mark.slow
def test_level_4_processes_nothing_before_the_edit():
    server.cache_splice = None
    server.cache_splice_level = 4
    # no checkpoint lies before the edit, so rolling back would mean starting over
    server.checkpoint_min_step = None
    server.start()
    complete(OLD)
    edited = complete(NEW)
    assert edited["cache_n"] >= n_tokens(HEAD)
    assert edited["splice_n"] > n_tokens(TAIL) // 2
    assert edited["cache_n"] + edited["splice_n"] + edited["prompt_n"] == n_tokens(NEW)


@pytest.mark.slow
def test_request_can_turn_splice_off():
    server.start()
    complete(OLD)
    edited = complete(NEW, n_cache_splice=0)
    assert edited["splice_n"] == 0
    assert edited["cache_n"] + edited["prompt_n"] == n_tokens(NEW)


@pytest.mark.slow
def test_checkpoints_keep_the_prefix_before_an_edit():
    server.cache_splice = 0
    server.start()
    complete(OLD)
    edited = complete(NEW)
    # the edit follows HEAD, and a checkpoint lies at most min step before it
    assert n_tokens(HEAD) - 2 * 64 < edited["cache_n"] < n_tokens(HEAD)
    assert edited["splice_n"] == 0

    # the checkpoint taken where the prompts diverged keeps the whole shared prefix next time
    again = complete(prompt("NOTE: the valve code will change tomorrow."))
    assert again["cache_n"] >= n_tokens(HEAD)


@pytest.mark.slow
def test_checkpoints_on_disk_behave_like_checkpoints_in_memory(tmp_path):
    global server
    server.start()
    complete(OLD)
    in_memory = complete(NEW)
    server.stop()

    server = hybrid()
    server.checkpoint_path = str(tmp_path)
    server.start()
    complete(OLD)
    assert len(os.listdir(tmp_path)) > 0
    on_disk = complete(NEW)
    for key in ("cache_n", "splice_n", "prompt_n"):
        assert on_disk[key] == in_memory[key]

    server.stop()
    assert os.listdir(tmp_path) == []


@pytest.mark.slow
def test_checkpoint_files_of_a_killed_server_are_removed_by_the_next(tmp_path):
    global server
    server.checkpoint_path = str(tmp_path)
    server.start()
    complete(OLD)
    server.process.kill()
    server.process.wait()
    left = [name for name in os.listdir(tmp_path) if name.endswith(".bin")]
    assert len(left) > 0

    server = hybrid()
    server.checkpoint_path = str(tmp_path)
    server.start()
    assert not set(left) & set(os.listdir(tmp_path))
    assert [name for name in os.listdir(tmp_path) if name.endswith(".lock")] != []

    server.stop()
    assert os.listdir(tmp_path) == []


def chat(user: str) -> dict:
    res = server.make_request("POST", "/chat/completions", data={
        "messages": [
            {"role": "system", "content": f"You answer questions about this log.\n{HEAD}"},
            {"role": "user", "content": user},
        ],
        "max_tokens": 8,
        "temperature": 0.0,
        "cache_prompt": True,
        "id_slot": 0,
    })
    assert res.status_code == 200
    return res.body


@pytest.mark.slow
def test_system_prompt_is_loaded_after_a_restart(tmp_path):
    global server
    # message boundaries come from the chat template
    server.jinja = True
    server.system_cache_path = str(tmp_path)
    server.start()
    first = chat("Which station reported in entry 3?")
    assert first["timings"]["cache_n"] == 0
    saved = os.listdir(tmp_path)
    assert len(saved) == 1
    reference = chat("Which station reported in entry 7?")
    server.stop()

    server = hybrid()
    server.jinja = True
    server.system_cache_path = str(tmp_path)
    server.start()
    loaded = chat("Which station reported in entry 7?")
    assert os.listdir(tmp_path) == saved
    # everything before the user message came from the file
    assert n_tokens(HEAD) < loaded["timings"]["cache_n"] == reference["timings"]["cache_n"]
    assert loaded["timings"]["prompt_n"] == reference["timings"]["prompt_n"]
    assert loaded["choices"][0]["message"]["content"] == reference["choices"][0]["message"]["content"]

    # a later edit can still return to the end of the system prompt
    again = chat("Which station reported in entry 9?")
    assert again["timings"]["cache_n"] >= loaded["timings"]["cache_n"]


def test_splice_is_ignored_without_recurrent_state():
    global server
    server = ServerPreset.tinyllama2()
    server.n_ctx = 2048
    server.n_slots = 1
    server.cache_splice = 6
    server.start()
    old = prompt(OLD_NOTE, lines("north", 8), lines("south", 12))
    new = prompt(NEW_NOTE, lines("north", 8), lines("south", 12))
    complete(old)
    edited = complete(new)
    assert edited["splice_n"] == 0
    assert edited["cache_n"] + edited["prompt_n"] == n_tokens(new)
