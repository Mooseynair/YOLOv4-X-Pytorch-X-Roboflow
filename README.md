# YOLOv4 X Pytorch X Roboflow

# Description

---

Aight you fuckers. This is the situation with this code and this process and all the blood sweat and tears that went into figuring this shit out.

### What are we trying to achieve?

Using a Windows 11 computer, we managed to get YOLOv4 working (using a Pytorch framework) with a local GPU and used Roboflow to assist in collating the data into train, test and validation folders. We used Ananconda to create a seperate environment to manage all our depedencies

# Step 1 - Clone the repo and make some file edits

---

### Step 1.1 - Clone repo

First create a new folder where you will be cloning the repo into. Let’s call it “Yolo_Roboflow”

Navigate to said folder (”Yolo_Roboflow”) and run the command

```bash
cd Yolo_Roboflow

git clone https://github.com/roboflow-ai/pytorch-YOLOv4.git
```

### Step 1.2 - Edit files

Inside the cloned folder “pytorch-YOLOv4” open up “train.py” and “dataset.py” and “requirements.txt” to make some edits

**train.py**
Change “.view” to “.reshape”

![Alt text](README-images/img1.png?raw=true)

![Alt text](README-images/img2.png?raw=true)

Add those three extra lines highlighted in green

![Alt text](README-images/img3.png?raw=true)

```python
my_device = torch.cuda.current_device()
my_device_name = torch.cuda.get_device_name(my_device)
logging.info(f'Device name {my_device_name}')
```

**dataset.py**

![Alt text](README-images/img4.png?raw=true)

**requirements.txt**

```bash
"requirements.txt"

numpy==1.22.3
tensorboardX==2.2
matplotlib==3.5.1
tqdm==4.64.0
easydict==1.9
Pillow==9.0.1
opencv==4.5.5
pycocotools==2.0.4

# might need to add conda install jupyter to this
# and even conda install -c conda-forge onnx
```

# Step 2 - Setup virtual environment and install dependenices

---

### Step 2.1 - Setup virtual environment and activate it

Create a new virtual environment using Anaconda Prompt (the anaconda terminal) using this command 

```python
conda create --name [name of virtual environemnt] [python version]

e.g.
conda create --name Yolov4_env python=3.8.13
```

Activate your virtual environment via the anaconda terminal using 

```bash
conda activate [name of virtual environemnt]

e.g.
conda activate Yolov4_env

```

### Step 2.2 -  Install dependencies

in the conda terminal, navigate to the folder where you cloned the repo and cd into pytorch-YOLOv4. From here we can now install our requirements.txt via the following command 

```bash
cd Yolo_Roboflow/pytorch-YOLOv4
conda install --file requirements.txt
```

You will also need to install this pytorch and cuda stuff from this link [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/). (see picture below for installation details)

![Alt text](README-images/img5.png?raw=true)

simply run the given command from the website in the conda terminal like so 

```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
```

# Step 3 - Download Yolov4 weights

---

Download yolov4 weights (yolov4.conv.137.pth) that have already been converted to pytorch via the link: [https://drive.google.com/uc?id=1fcbR0bWzYfIEdLJPzOsn4R5mlvR6IQyA](https://drive.google.com/uc?id=1fcbR0bWzYfIEdLJPzOsn4R5mlvR6IQyA)

Place the downlaoded weights in your cfg folder in the path “Yolo_Roboflow/pytorch-YOLOv4/cfg

# Step 4 - Train - Google colab vs Local

---

## Step 4.1 Using google colab to train

### Step 4.1.1

This google colab link shows you how to use Roboflow to train your model on a custom dataset: [https://colab.research.google.com/drive/1TWRVB0SxxEd4P9h6L0ftNLoEKbUInpkj#scrollTo=mIlhqP2S57Ub](https://colab.research.google.com/drive/1b08y_nUYv5UtDY211NFfINY7Hy_pgZDt#scrollTo=091QOGGihsuV)

At the end of the training you should be able to just donwload the weights you need following the file explorer GUI already in the collab environment on the left hand side of the screen. 

## Step 4.2 Using local GPU to train

### Step 4.2.1 - Mimicking the Colab environment on our local computer

Sinve we’ve already cloned the repo and installed the necessary dependencies onto our local computer/environment the only thing we need to do now is download the custom dataset from Roboflow. 

go to [Roboflow.com](http://Roboflow.com) → sign in → click “Universe” tab on the top ribbon → search for whataver dataset you want → click “Donwload this dataset” → click on “download zip to computer”.

Once downloaded simply extract the contents into the folder where the repo was initially cloned (in our case that’s Yolo_Roboflow) and it should look something like this 

![Alt text](README-images/img6.png?raw=true)

While in the Yolo_Roboflow folder, run these commands using wsl

```bash
cp train/_annotations.txt train/train.txt
cp train/_annotations.txt train.txt
mkdir data
cp valid/_annotations.txt data/val.txt
cp valid/*.jpg train/
```

then train the data (change argument number for -classes  to its respective number)

```bash

#start training
#-b batch size (you should keep this low (2-4) for training to work properly)
#-s number of subdivisions in the batch, this was more relevant for the darknet framework
#-l learning rate
#-g direct training to the GPU device
#pretrained invoke the pretrained weights that we downloaded above
#classes - number of classes
#dir - where the training data is
#epoch - how long to train for

python ./pytorch-YOLOv4/train.py -b 2 -s 1 -l 0.001 -g 0 -pretrained ./pytorch-YOLOv4/cfg/yolov4.conv.137.pth -classes 4 -dir ./train -epochs 100
```

model weights should be saved in a folder called checkpoints

# Step 5 - Inference

```python
python demo_pytorch2onnx.py <weight_file> <image_path> <batch_size> <n_classes> <IN_IMAGE_H> <IN_IMAGE_W>

e.g.
python ./pytorch-YOLOv4/demo_pytorch2onnx.py ./checkpoints/Yolov4_epoch80.pth {img_path} 8 {num_classes} 608 608

```# YOLOv4-X-Pytorch-X-Roboflow
# YOLOv4-X-Pytorch-X-Roboflow
