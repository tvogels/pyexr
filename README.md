
# PyEXR

A simple EXR IO-library for Python that simplifies the use of [OpenEXR](https://github.com/jamesbowman/openexrpython).

### Installation

~~~bash
pip install git+git://github.com/tvogels/pyexr.git
~~~

### Reading

__Simple files__

~~~python
import pyexr

file = pyexr.open("color.exr")

file.channels               # [R, G, B]
file.width                  # 1280
file.height                 # 720
file.channel_precision["R"] # pyexr.FLOAT

img = file.get()                         # (720,1280,3) np.float32 array
img = file.get(type=pyexr.HALF)          # (720,1280,3) np.float16 array
red = file.channel("R")                  # (720,1280,1) np.float32 array
red = file.channel("R", type=pyexr.HALF) # (720,1280,1) np.float16 array

~~~


__Fat / Multi-channel EXRs__

~~~python
import pyexr

file = pyexr.open("multi-channel.exr")

file.channels               # [R, G, B, A, Variance.R, Variance.G, Variance.B]
file.width                  # 1280
file.height                 # 720

all = file.get()            # (720,1280,7) np.float32 array (R,G,B,A,Var..)
var = file.get("Variance")  # (720,1280,3) np.float32 array
col = file.get("default")   # (720,1280,4) np.float32 array (R,G,B,A)
file.channel_map['default'] # ['R','G','B','A']

var_r = file.channel("Variance.R") # (720,1280,3) np.float32 array
~~~


### One line reading

~~~python
import pyexr

# 'color.exr' contains R, G, B
img = pyexr.read("color.exr")                  # (720,1280,3) np.float32 array
img = pyexr.read("color.exr", type=pyexr.HALF) # (720,1280,3) np.float16 array

# 'multi-channel.exr' contains R, G, B, A, Variance.R, Variance.G, Variance.B
all = pyexr.read("multi-channel.exr")             # (720,1280,7) np array
col = pyexr.read("multi-channel.exr", "default")  # (720,1280,4) np array
var = pyexr.read("multi-channel.exr", "Variance") # (720,1280,3) np array

col, var = pyexr.read("multi-channel.exr", ["default", "Variance"])
col, var = pyexr.read("multi-channel.exr", ["default", "Variance"], type=pyexr.HALF)
col, var = pyexr.read("multi-channel.exr", ["default", "Variance"], type=[pyexr.HALF, pyexr.FLOAT])
~~~

### Writing

You can write write a matrix to EXR without specifying channel names. Default channel names will then be used:

| *#* columns | default    |
| ----------- | ---------- |
| 1 channel   | Z          |
| 2 channels  | X, Y       |
| 3 channels  | R, G, B    |
| 4 channels  | R, G, B, A |


~~~python
import pyexr

depth  # (720,1280) np.float16 array
color  # (720,180,3) np.float32 array
normal # (720,180,3) np.float32 array

exr.write("out.exr", depth)            # one FLOAT channel: Z
exr.write("out.exr", color)            # three FLOAT channels: R, G, B
exr.write("out.exr", normal, 
          channel_names=['X','Y','Z']) # three FLOAT channels: X, Y, Z

~~~

__Writing Fat EXRs__

~~~python
import pyexr

depth    # (720,1280) np.float16 array
color    # (720,180,3) np.float32 array
variance # (720,180,3) np.float32 array

data = {'default': color, 'Depth': depth, 'Variance': variance} # default is a reserved name

exr.write("out.exr", data) # channels R, G, B, Depth.Z, Variance.(R,G,B)

# Full customization:
exr.write(
    "out.exr", 
    data, 
    type = {'default': exr.HALF},
    channel_names = {'Depth': ['Q']}
) # channels R, G, B, Depth.Z, Variance.(R,G,B)

~~~
