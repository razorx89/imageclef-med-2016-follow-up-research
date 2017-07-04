ImageCLEFmed 2016 Follow-Up Research
====================================

This repository contains source code to replicate experiments done for the 
*ImageCLEFmed 2016 Subfigure Classification* dataset. An overall overview of
the dataset and task can be found in [1]. Details about the experiments can be
found the paper mentioned below. 

For all of the experiments [Tensorflow](https://www.tensorflow.org/) [2] version 
r11 was used. In order to fine-tune pre-trained network architectures, the 
Tensorflow [slim model repository](https://github.com/tensorflow/models/tree/master/slim)
was utilized with a [snapshot](https://github.com/tensorflow/models/tree/a315e5681d9cfee90f3adba460fd63b29ad886f9)
of Tensorflow r11 compatible code.

Usage
-----

1. Modify image dataset paths in: `slim/datasets/convert_imageclef_med_2016.py`
2. Download pretrained models and build databases: `./prepare.sh`
3. Train models: `./train.sh`
4. Evaluate models: `./evaluate.sh`

If you want to train different network architectures or configurations, then 
you need to modify the variables in `train.sh` and `evaluate.sh`.

Citation
--------
If you plan to use this work for your own research, then please cite our paper 
at the CLEF 2017 conference: 

> S. Koitka, C.M. Friedrich, "Optimized Convolutional Neural Network Ensembles 
> for Medical Subfigure Classification". In: Experimental IR Meets 
> Multilinguality, Multimodality, and Interaction. Proceedings of the 8th 
> International Conference of the CLEF Association, CLEF 2017, Dublin, Ireland, 
> September 11-14, 2017. Lecture Notes of Computer Science (LNCS), vol. 10456, 
> pp. ??-??, Springer Verlag (2017)

__LaTeX BibTex:__
```
@InProceedings{Koitka.Friedrich2017,
  author    = {Sven Koitka and Christoph M. Friedrich},
  title     = {Optimized Convolutional Neural Network Ensembles for Medical Subfigure Classification},
  booktitle = {Experimental IR Meets Multilinguality, Multimodality, and Interaction. Proceedings of the 8th International Conference of the CLEF Association, CLEF 2017, Dublin, Ireland, September 11-14, 2017},
  year      = {2017},
  editor    = {Gareth J. F. Jones, Séamus Lawless, Julio Gonzalo, Liadh Kelly, Lorraine Goeuriot, Thomas Mandl, Linda Cappellato, Nicola Ferro},
  volume    = {10456},
  series    = {Lecture Notes of Computer Science (LNCS)},
  pages     = {??--??},
  publisher = {Springer Verlag},
}
```

References
----------
[1] A. García Seco de Herrera, R. Schaer, S. Bromuri, H. Müller: "Overview of 
the ImageCLEF 2016 Medical Task". CLEF2016 Working Notes, CEUR-WS.org <ceur-ws.org>, vol. 1609, 2016.

[2] Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo,
Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy Davis,
Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow,
Andrew Harp, Geoffrey Irving, Michael Isard, Rafal Jozefowicz, Yangqing Jia,
Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dan Mané, Mike Schuster,
Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Jonathon Shlens,
Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker,
Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viégas,
Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke,
Yuan Yu, and Xiaoqiang Zheng.
TensorFlow: Large-scale machine learning on heterogeneous systems, 2015. Software available from tensorflow.org.
