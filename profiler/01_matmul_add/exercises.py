import json, os
OUT = os.environ.get("EX_OUT", "/tmp")
os.makedirs(OUT, exist_ok=True)
import torch
from torch.profiler import profile, ProfilerActivity, schedule

ACT = [ProfilerActivity.CPU, ProfilerActivity.CUDA]


def prep(n=64, dtype=torch.bfloat16):
    x = torch.randn(n, n, device="cuda", dtype=dtype)
    w = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    def fn(x, w, b): return torch.add(torch.matmul(x, w), b)
    return x, w, b, fn


def rows(p, filt=None):
    out = []
    for e in p.key_averages():
        if filt is None or filt(e.key):
            out.append((e.key, e.self_cpu_time_total, e.cpu_time_total,
                        e.device_time_total, e.count))
    return out


def q1():
    print("=" * 70)
    print("Q1  标签换回固定字符串 'matmul_add'，表里行数变化吗？")
    x, w, b, fn = prep()
    def step():
        with torch.profiler.record_function("matmul_add"):
            return fn(x, w, b)
    for _ in range(5): step()
    torch.cuda.synchronize()
    with profile(activities=ACT, schedule=schedule(wait=1, warmup=1, active=5, repeat=1)) as p:
        for _ in range(7):
            step(); p.step()
    r = rows(p, lambda k: "matmul_add" in k)
    print(f"  命中行数 = {len(r)}")
    for k, sc, tc, dt, c in r:
        print(f"    {k!r:<18} SelfCPU={sc:8.1f}us  CPUtotal={tc:8.1f}us  CUDA={dt:8.1f}us  calls={c}")


def q2():
    print("=" * 70)
    print("Q2  record_shapes=True，aten::mm 那行名字变成什么？")
    x, w, b, fn = prep(1024)
    def step():
        with torch.profiler.record_function("matmul_add_(1024, 1024)"):
            return fn(x, w, b)
    for _ in range(5): step()
    torch.cuda.synchronize()
    with profile(activities=ACT, schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
                 record_shapes=True) as p:
        for _ in range(5):
            step(); p.step()
    for k, sc, tc, dt, c in rows(p, lambda k: "mm" in k or "add" in k or "matmul" in k):
        print(f"    {k!r:<46} CUDA_avg={dt / max(c,1):8.3f}us  calls={c}")
    p.export_chrome_trace(OUT + "/ex_q2.json")
    ev = json.load(open(OUT + "/ex_q2.json"))["traceEvents"]
    with_dims = [e for e in ev if "Input Dims" in (e.get("args") or {})]
    print(f"  → trace json 里带 'Input Dims' 的事件数 = {len(with_dims)}")
    for e in with_dims[:4]:
        print(f"      {e['name'][:34]:<34} args={e['args']}")
    print("  → key_averages 里 aten::mm 的 input_shapes 属性：")
    for e in p.key_averages():
        if e.key == "aten::mm":
            print(f"      input_shapes = {e.input_shapes!r}")


def q3():
    print("=" * 70)
    print("Q3  schedule 改 wait=0, warmup=0, active=5：循环次数改多少？前几步有什么不同？")
    for wt, wm in [(1, 1), (0, 0)]:
        x, w, b, fn = prep()
        def step():
            with torch.profiler.record_function("matmul_add_(64, 64)"):
                return fn(x, w, b)
        for _ in range(5): step()
        torch.cuda.synchronize()
        loop = wt + wm + 5                      # 一轮 = wait + warmup + active
        with profile(activities=ACT, schedule=schedule(wait=wt, warmup=wm, active=5, repeat=1)) as p:
            for _ in range(loop):
                step(); p.step()
        mm = [r for r in rows(p, lambda k: k == "aten::mm")]
        ad = [r for r in rows(p, lambda k: k == "aten::add")]
        path = f"{OUT}/ex_q3_{wt}{wm}.json"
        p.export_chrome_trace(path)
        ev = json.load(open(path))["traceEvents"]
        steps = sorted(e["name"] for e in ev if e.get("cat") == "user_annotation"
                       and e["name"].startswith("ProfilerStep"))
        ann = sorted([e for e in ev if e.get("cat") == "user_annotation"
                      and e["name"].startswith("matmul_add")], key=lambda e: e["ts"])
        print(f"  wait={wt} warmup={wm} → 循环 {loop} 次，记录到 {len(ann)} 步: {steps}")
        if mm:
            print(f"    aten::mm  CUDA_avg = {mm[0][3]/mm[0][4]:8.3f}us   calls={mm[0][4]}")
        if ad:
            print(f"    aten::add CUDA_avg = {ad[0][3]/ad[0][4]:8.3f}us")
        print(f"    每步 CPU annotation: {[round(e['dur'],1) for e in ann]}")


def q4():
    print("=" * 70)
    print("Q4  嵌套 record_function('outer') 套住 step()：层级和时间线")
    x, w, b, fn = prep()
    def step():
        with torch.profiler.record_function("outer"):
            with torch.profiler.record_function("matmul_add_(64, 64)"):
                return fn(x, w, b)
    for _ in range(5): step()
    torch.cuda.synchronize()
    with profile(activities=ACT, schedule=schedule(wait=1, warmup=1, active=5, repeat=1)) as p:
        for _ in range(7):
            step(); p.step()
    p.export_chrome_trace(OUT + "/ex_q4.json")
    ev = json.load(open(OUT + "/ex_q4.json"))["traceEvents"]
    import collections
    cnt = collections.Counter(e["name"] for e in ev if e.get("cat") == "user_annotation")
    print("  表里（含 outer / matmul_add_ 的行）：")
    for k, sc, tc, dt, c in sorted(rows(p, lambda k: k in ("outer", "matmul_add_(64, 64)")),
                                   key=lambda r: -r[2]):
        print(f"    {k!r:<20} SelfCPU={sc:8.1f}us  CPUtotal={tc:8.1f}us  CUDA={dt:8.1f}us  calls={c}")
    print(f"  时间线上 user_annotation 事件: {dict(cnt)}")


def q5():
    print("=" * 70)
    print("Q5  with_stack=True，trace json 从多大涨到多大？")
    for ws in (False, True):
        x, w, b, fn = prep(1024)
        def step():
            with torch.profiler.record_function("matmul_add_(1024, 1024)"):
                return fn(x, w, b)
        for _ in range(5): step()
        torch.cuda.synchronize()
        with profile(activities=ACT, schedule=schedule(wait=1, warmup=1, active=10, repeat=1),
                     with_stack=ws) as p:
            for _ in range(12):
                step(); p.step()
        path = f"{OUT}/ex_q5_stack_{int(ws)}.json"
        p.export_chrome_trace(path)
        sz = os.path.getsize(path)
        print(f"  with_stack={str(ws):<5} → {sz/1024:8.1f} KB   ({path})")


def q6():
    print("=" * 70)
    print("Q6  prof.step() 挪到循环外：空报告")
    x, w, b, fn = prep()
    def step():
        with torch.profiler.record_function("matmul_add_(64, 64)"):
            return fn(x, w, b)
    for _ in range(5): step()
    torch.cuda.synchronize()
    with profile(activities=ACT, schedule=schedule(wait=1, warmup=1, active=5, repeat=1)) as p:
        for _ in range(7):
            step()
        p.step()                      # ★ 挪到循环外
    print(f"  事件总数 = {len(p.events())}")
    print(p.key_averages().table(sort_by="cuda_time_total", row_limit=6))


def q7():
    print("=" * 70)
    print("Q7  sort_by='self_cpu_time_total'，64×64 下哪行排第一？")
    x, w, b, fn = prep()
    def step():
        with torch.profiler.record_function("matmul_add_(64, 64)"):
            return fn(x, w, b)
    for _ in range(10): step()
    torch.cuda.synchronize()
    with profile(activities=ACT, schedule=schedule(wait=1, warmup=1, active=20, repeat=1)) as p:
        for _ in range(22):
            step(); p.step()
    print("  按 self_cpu_time_total 排序 Top 6：")
    print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=6))


if __name__ == "__main__":
    for f in (q1, q2, q3, q4, q5, q6, q7):
        f()
        print()
