import subprocess

print("ResNet18_FULL...")
subprocess.run([
    "python", 
    "train.py", 
    "--ResConnection", "1",
    "--BatchNorm", "1", 
    "--ModelName", "ResNet18_FULL",
    "--Depth", "18",
    "--Seed", "42",
])

print("ResNet18_Nbn...")
subprocess.run([
    "python", 
    "train.py", 
    "--ResConnection", "1",
    "--BatchNorm", "0", 
    "--ModelName", "ResNet18_Nbn",
    "--Depth", "18",
    "--Seed", "42",
])

print("ResNet18_Nres...")
subprocess.run([
    "python", 
    "train.py", 
    "--ResConnection", "0",
    "--BatchNorm", "1", 
    "--ModelName", "ResNet18_Nres",
    "--Depth", "18",
    "--Seed", "42",
])

print("ResNet18_Nboth...")
subprocess.run([
    "python", 
    "train.py", 
    "--ResConnection", "0",
    "--BatchNorm", "0", 
    "--ModelName", "ResNet18_Nboth",
    "--Depth", "18",
    "--Seed", "42",
])

print("ResNet50_FULL...")
subprocess.run([
    "python", 
    "train.py", 
    "--ResConnection", "1",
    "--BatchNorm", "1", 
    "--ModelName", "ResNet50_FULL",
    "--Depth", "50",
    "--Seed", "42",
])

print("ResNet50_Nbn...")
subprocess.run([
    "python", 
    "train.py", 
    "--ResConnection", "1",
    "--BatchNorm", "0", 
    "--ModelName", "ResNet50_Nbn",
    "--Depth", "50",
    "--Seed", "42",
])

print("ResNet50_Nres...")
subprocess.run([
    "python", 
    "train.py", 
    "--ResConnection", "0",
    "--BatchNorm", "1", 
    "--ModelName", "ResNet50_Nres",
    "--Depth", "50",
    "--Seed", "42",
])

print("ResNet50_Nboth...")
subprocess.run([
    "python", 
    "train.py", 
    "--ResConnection", "0",
    "--BatchNorm", "0", 
    "--ModelName", "ResNet50_Nboth",
    "--Depth", "50",
    "--Seed", "42",
])