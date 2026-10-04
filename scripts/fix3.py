path = "/dev/shm/alpha-verify-step7.ts"
with open(path, "rb") as f:
    data = f.read()
lines = data.split(b"\n")
fixed = []
for line in lines:
    if b"`);" in line:
        idx = line.rfind(b"`);")
        if idx != -1:
            line = line[:idx] + b"`" + b"," + line[idx+3:]
    fixed.append(line)
with open(path, "wb") as f:
    f.write(b"\n".join(fixed))
print("Fixed")
