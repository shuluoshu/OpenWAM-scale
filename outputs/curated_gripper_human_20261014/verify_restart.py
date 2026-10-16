"""Verify the running human-supervised experiment and both TensorBoards."""
import hashlib
import json
import math
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request

OUT = Path(__file__).resolve().parent
NAME = "gripper_human_lr0.0001_b320"
BASE = "http://127.0.0.1:18083"


def fetch(base, path, **params):
    url = base + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=20) as response:
        assert response.status == 200
        return json.load(response)


status = json.loads((OUT / "status.json").read_text())
assert status["phase"] == "training", status
for key in ("pid", "supervisor_pid", "tensorboard_pid"):
    proc = Path("/proc") / str(status[key])
    assert proc.exists() and (proc / "stat").read_text().split()[2] != "Z", key
restart = json.loads((OUT / "restart.json").read_text())
for item in restart["preserved_event_prefixes"]:
    with Path(item["path"]).open("rb") as handle:
        assert hashlib.sha256(handle.read(item["bytes"])).hexdigest() == item["sha256"]
contract = json.loads((OUT / NAME / "label_contract.json").read_text())
assert contract["gripper_normalization"] == "clip_0_1"
assert set(contract["human_families"]) == {"LightWheel", "mecka", "egodex", "egoverse"}
tags = fetch(BASE, "/data/plugin/scalars/tags")
assert NAME in tags
wanted = ["train/loss_total", "train/loss_action", "train/loss_video",
          "eval_fixed_noise/clean_gripper_mse", "eval_fixed_noise/clean_joint_mse",
          "eval_fixed_noise/loss_fk_position"]
wanted += [f"eval_dataset/{clip}/clean_gripper_mse" for clip in
           ["04_LightWheel", "12_egodex", "16_egoverse", "24_mecka"]]
checks = {}
for tag in wanted:
    assert tag in tags[NAME], tag
    points = fetch(BASE, "/data/plugin/scalars/scalars", run=NAME, tag=tag)
    assert points and all(math.isfinite(p[2]) for p in points), tag
    checks[tag] = dict(points=len(points), last_step=points[-1][1], last_value=points[-1][2])
assert checks["train/loss_total"]["last_step"] >= 1
shared_tags = fetch("http://127.0.0.1:18082", "/data/plugin/scalars/tags")
assert NAME in shared_tags, "New run absent from old comparison TensorBoard"
assert {"main_lr0.0001_b320", "gripper01_lr0.0001_b320", "gripper_cmd_lr0.0001_b320"} <= set(shared_tags)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


gateway = "https://897628-proxy-18083.dsw-gateway-cn-shanghai.data.aliyuncs.com/"
try:
    with urllib.request.build_opener(NoRedirect).open(gateway, timeout=30) as response:
        gateway_result = dict(status=response.status)
except urllib.error.HTTPError as error:
    location = urllib.parse.urlsplit(error.headers.get("Location", ""))
    gateway_result = dict(status=error.code, redirect_host=location.netloc, redirect_path=location.path)
except urllib.error.URLError as error:
    gateway_result = dict(error=str(error.reason))
report = dict(all_passed=True, new_tensorboard_http=200, gateway=gateway_result,
    old_event_prefixes_preserved=True, run=NAME, previous_last_step=status["previous_last_step"],
    scalar_checks=checks, comparison_runs=sorted(shared_tags), supervisor_pid=status["supervisor_pid"],
    torchrun_pid=status["pid"], tensorboard_pid=status["tensorboard_pid"], contract=contract)
(OUT / "restart_validation.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
