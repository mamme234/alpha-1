path = "/dev/shm/alpha-verify-step7.ts"
with open(path, "rb") as f:
    data = f.read()
lines = data.split(b"\n")
print("line28 repr:", repr(lines[27]))
print("has b'`);':", b"`);" in lines[27])
idx = lines[27].rfind(b"`);")
print("idx:", idx)
print("line28[:idx]:", repr(lines[27][:idx]))
print("line28[idx+3:]:", repr(lines[27][idx+3:]))
