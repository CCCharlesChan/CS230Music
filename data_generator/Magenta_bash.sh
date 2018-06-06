source activate magenta
melody_rnn_generate --config='attention_rnn' --bundle_file="/Users/charleschen/Documents/Courses/CS230/Project/Code/test/data_generator/attention_rnn.mag" --output_dir="/Users/charleschen/Documents/Courses/CS230/Project/Code/test/data_generator/magenta" --num_outputs=1 --num_steps=2048 --primer_melody="[60]"
source deactivate magenta