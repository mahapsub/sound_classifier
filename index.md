# Sound Classification Blog
### By: Subrat Mahapatra
Contact: smahapat7@gmail.com
EECS/MSAI 349, Fall 2018, Northwestern University

## The Motivation

This project starts off exploring the space of general purpose audio tagging, which has been generally overlooked when it comes to problems like computer vision. The motivation for this project is to understand the our enviornment in a way that images cannot capture. For example a clip of sounds at club with poor lighting could indicate popularity much better than analysis on low-lighting images can provide. Tagging sound clips also provides a perspective on information that is collected passively. For example, a grinder in the background of a coffee shop might indicate that the coffee is likely brewed fresh.  In this example understanding these details about a coffee shop or a club might be a significant indicator of sales across a chain of coffee shops. 

Understanding the environment through sound gives us a level of access that might surpass the limits of what computer vision can provide. For example, the sophistication of any and all vision models require all the objects  in the environment to be not occluded by larger objects. This sort of problem is not really seen in traditional sound classification situations. Sound classification however does come with its own share of problems. Sounds picked up by a microphone might be mixed with several signals. A good model should not only be able to tag sounds with the appropriate label, but be able to differentiate between the different sounds that a microphone might pick up. 

## The Problem Task
The purpose of this project is to develop models that can accurately tag various sounds that can be found in the environment. More specificially, to be able to tag a dataset of sounds with the appropriate lablels. The audio clips are from the Freesound General-Purpose Audio Tagging Challenge hosted by Kaggle. These clips are on average 6 seconds of audio along with their corresponding label. These clips only contain one type of sound, and one label for each file. This project is a critcal component in general audio tagging because one specific sounds can be differentiated from each other, then future work could focus on denoising mixed sound signals. 

## The Data
![Image](images/data_visual.png?raw=true)

Other Key Points:
- Resolution bit depth of 16
- Sampling rate of 44.1 kHz
- Pulse code modulated

![Image](images/figures/Saxophone_graph.png?raw=true)
Saxophone

![Image](images/figures/Violin_or_fiddle_graph.png?raw=true)
Violin_or_fiddle

## Classifying between a Saxophone and a Violin/Fiddle.

I used a neural netowrk model to see if I could mimick the success of computer vision models. I started off with a 1D convolution network that I took from Zafar's Kernel found on Kaggle (Link below) which was configured for classifying labels for mixed sounds and modified the code. Zafar then moves on to the 2D convolutional model, but I decided to focus my research on trying to increase the 1D accuracies. I then noticed that the model was overfitting quick quickly so I modified some of the layers by increasing dropout. I then changed the structure of the layers and experiemented with learning rate, sound duration, and activation functions. The accuracies that I got were from 2-fold cross validation because I was restricted with computational power. These accuracies will most likely change to reflect the true accuracy with something close to 10-fold cross validation.

## Model Selection

![Image](images/figures/graph_acc.png?raw=true)


![Image](images/figures/table_val.png?raw=true)

These figures demonstrate the ZeroR result as well as the base model that was taken from Zafar's kernel. It additionally shows test accuracies from the various models that were explored.






## Problems
1. The training data has a specific sound per entry but the test data contains a mixture of sounds.
2. The test data would be a good test set for a general classifier but is not a good test set for my
   subproblem. This means that the data that I can use to train my examples decreases by 30% as that is
   allotted to a dedicated test set.
3. 


### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/mahapsub/sound_classifier/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Reference

Thank's so much for Zafar's clear and concise explanations when it comes to dealing with sound!
https://www.kaggle.com/fizzbuzz/beginner-s-guide-to-audio-data

