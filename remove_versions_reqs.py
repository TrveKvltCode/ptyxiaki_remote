
dependencies = []
with open("test.txt" , 'r', encoding="utf-8") as file:
    dependencies = file.readlines()

with open("requirements.txt", "w") as file2:
    for dependency in dependencies:
        dependency = dependency.split("==")[0]+"\n"
        file2.write(dependency)