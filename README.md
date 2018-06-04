# Fine-tuning a model from an existing checkpoint
This section wil describe how to fine-tune your own model from the existing checkpoint of Inception V3. This tutorial is referenced 
from https://github.com/tensorflow/models/tree/master/research/slim#fine-tuning-a-model-from-an-existing-checkpoint with some changes
to train your own model.
The example training dataset is hospital dataset, which is used to recognize doctor and patient.

# I/ Prepare dataset to train
Create training directory ./data/hospital/hospital_photos. Then download hospital dataset photos to train. Each category is stored in a separated folder whose name is "class" to describe. (e.g. doctor, patient).
Save these folders in ./data/hospital/hospital_photos/
Dataset storage will be in that structure:
```
data
└── hospital
    └── hospital_photos
        ├── doctor
        └── patient
```

# II/ Modify set-up file to convert dataset to TRRecord format
1. Copy datasets/convert_poses.py to datasets/convert_hospital.py
```ruby
cp datasets/convert_poses.py datasets/convert_hospital.py
```
2. Open file datasets/convert_hospital.py, edit **__NUM_VALIDATION_** variable to desired number of validation photos.
This number depends on the size of your dataset. Then replace all words "poses" by "hospital" in this file.
3. Open file download_and_convert_data.py
- Add **_from datasets import convert_hospital_**
- Add **_hospital_** to **_tf.app.flags.DEFINE_string_**
- Add command **_convert_hospital.run(FLAGS.dataset_dir)_** to "if" in **_main(_)**

# III/ Convert to TFRecord Format
For each dataset, we'll need to download the raw data and convert it to TensorFlow's native TFRecord format. Each TFRecord contains a TF-Example protocol buffer.
```ruby
python3 download_and_convert_data.py --dataset_name=hospital --dataset_dir=./data/hospital
```
# IV/ Modify set-up file to train model
1. Copy datasets/poses.py to datasets/hospital.py.
```ruby
cp ./datasets/poses.py ./datasets/hospital.py
```
2. Open file datasets/hospital.py, 
- Edit **_SPLITS_TO_SIZES_** to the number of photos used for training and validation.
- Edit **__NUM_CLASSES_** to 2 (because there are 2 classes: doctor and patient). 
- Edit **__FILE_PATTERN_** to **_hospital_%s_*.tfrecord_**
- Replace all words "poses" by "hospital"
3. Open file datasets/dataset_factory.py
- Add **_from datasets import hospital_**
- Add **_'hospital': hospital,_** to datasets_map

# V/ Download Checkpoint of pre-trained model of Inception V3 and fine-tune your own model
1. Download Checkpoint of InceptionV3 pre-trained model
```ruby
mkdir ./my_checkpoints
wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
tar -xvf inception_v3_2016_08_28.tar.gz
mv inception_v3.ckpt ./my_checkpoints/
rm inception_v3_2016_08_28.tar.gz
```
2. Fine-tune your own model
To indicate a checkpoint from which to fine-tune, we'll call training with the --checkpoint_path flag and assign it an absolute path to a checkpoint file.

When fine-tuning a model, we need to be careful about restoring checkpoint weights. In particular, when we fine-tune a model on a new task with a different number of output labels, we won't be able restore the final logits (classifier) layer. For this, we'll use the --checkpoint_exclude_scopes flag. This flag hinders certain variables from being loaded. When fine-tuning on a classification task using a different number of classes than the trained model, the new model will have a final 'logits' layer whose dimensions differ from the pre-trained model. For example, if fine-tuning an ImageNet-trained model on Hospital, the pre-trained logits layer will have dimensions [2048 x 1001] but our new logits layer will have dimensions [2048 x 2]. Consequently, this flag indicates to TF-Slim to avoid loading these weights from the checkpoint.

Keep in mind that warm-starting from a checkpoint affects the model's weights only during the initialization of the model. Once a model has started training, a new checkpoint will be created in --train_dir. If the fine-tuning training is stopped and restarted, this new checkpoint will be the one from which weights are restored and not the --checkpoint_path. Consequently, the flags --checkpoint_path and --checkpoint_exclude_scopes are only used during the 0-th global step (model initialization). Typically for fine-tuning one only want train a sub-set of layers, so the flag --trainable_scopes allows to specify which subsets of layers should trained, the rest would remain frozen.

Below we give an example of fine-tuning inception-v3 on Hos, inception_v3 was trained on ImageNet with 1000 class labels, but the Hospital dataset only have 2 classes. Since the dataset is quite small we will only train the new layers.
```ruby
python3 train_image_classifier.py \
    --train_dir=./hospital_models/inception_v3 \
    --dataset_dir=./data/hospital \
    --dataset_name=hospital \
    --dataset_split_name=train \
    --model_name=inception_v3 \
    --checkpoint_path=my_checkpoints/inception_v3.ckpt \
    --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
    --trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
    --max_number_of_steps=5000 \
    --batch_size=32 \
    --learning_rate=0.01 \
    --learning_rate_decay_type=fixed \
    --save_interval_secs=60 \
    --save_summaries_secs=60 \
    --log_every_n_steps=100 \
    --optimizer=rmsprop \
    --weight_decay=0.00004
```
For more information about Gradient Descent optimizer algorithm, you can refer at: http://ruder.io/optimizing-gradient-descent/index.html

# VI/ Evaluating performance of a model
To evaluate the performance of a model, you can use the eval_image_classifier.py script, as shown below.
```ruby
 python eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=./hospital_models/inception_v3/model.ckpt-5000 \
    --dataset_dir=./data/hospital \
    --dataset_name=hospital \
    --dataset_split_name=validation \
    --model_name=inception_v3
```
# VII/ Exporting the Inference Graph
Saves out a GraphDef containing the architecture of the model.
To use it with a model name defined by slim, run:
```ruby
python export_inference_graph.py \
  --alsologtostderr \
  --model_name=inception_v3 \
  --output_file=./hospital_icpv3_inf_graph.pb \
  --dataset_name=hospital
```
# VIII/ Freezing the exported Graph
  If you then want to use the resulting model with your own or pretrained checkpoints as part of a mobile model, you can run freeze_graph to get a graph def with the variables inlined as constants using:
  ```ruby
  python3 freeze_graph.py \
  --input_graph=./hospital_icpv3_inf_graph.pb \
  --input_checkpoint=./hospital_models/inception_v3/model.ckpt-5000 \
  --input_binary=true \
  --output_graph=./hospital_icpv3_frz_graph.pb \
  --output_node_names=InceptionV3/Predictions/Reshape_1
  ```
# IX/ Test your model
Run the code below:
```ruby
python3 label_image.py \
--graph=./hospital_icpv3_frz_graph.pb \
--labels=./data/hospital/labels.txt \
--image=<PATH_TO_IMAGE>
```
 

