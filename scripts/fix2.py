path = "/dev/shm/alpha-verify-step7.ts"
with open(path, "r") as f:
    content = f.read()

lines = content.split("\n")
fixed = []
for line in lines:
    if "`);" in line:
        idx = line.rfind("`);")
        if idx != -1:
            line = line[:idx] + "`" + "," + line[idx+3:]
    fixed.append(line)

with open(path, "w") as f:
    f.write("\n".join(fixed))
print("Fixed")
