import os
import re

TEMPLATE_TOKEN = "#TEMPLATE"
def get_path_from_token(token):
    return re.findall(r"[a-zA-Z/0-9_\-\.]+", token)[0] #select all characters possible in a file path description

def file_rename(f):
    pass

file_path = os.path.realpath(__file__)
template_dir = os.path.realpath(os.path.join(file_path, "..", "template"))

src_path = os.path.abspath(get_path_from_token(input().strip()))
print(template_dir)

with open(src_path, "r") as f:
    content = f.readlines()
    path_new = os.path.realpath(os.path.join(template_dir, os.path.basename(src_path)))

    with open(path_new, "w") as f_new:
        is_skipping = False
        for line in content:
            if TEMPLATE_TOKEN in line:
                if is_skipping:
                    indentation = line.index(TEMPLATE_TOKEN)
                    f_new.write(" " * indentation + "..." + "\n")
                is_skipping = not is_skipping
                continue
            if not is_skipping:
                f_new.write(line)
