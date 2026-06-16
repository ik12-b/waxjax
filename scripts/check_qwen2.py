import runpy, traceback
try:
    ns = runpy.run_path('safejax/architectures/qwen2.py')
    print('executed, keys=', [k for k in ns.keys() if not k.startswith('__')])
except Exception:
    traceback.print_exc()
