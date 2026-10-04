path = "/dev/shm/alpha-verify-step7.ts"
with open(path, "r") as f:
    content = f.read()

def fix(line):
    if "`);" in line:
        idx = line.rfind("`);")
        if idx != -1:
            # Replace `);` (3 chars) with `, (2 chars), dropping the `)
            line = line[:idx] + "`" + "," + line[idx+3:]
    return line

lines = content.split("\n")
fixed = [fix(l) for l in lines]

with open(path, "w") as f:
    f.write("\n".join(fixed))

print("Fixed")
