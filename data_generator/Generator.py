
# coding: utf-8

# In[63]:


import subprocess
import os.path as op
import numpy as np
from os import listdir


# In[70]:


working_dir = '/Users/charleschen/Documents/Courses/CS230/Project/Code/test/data_generator/'
magenta_dir = 'magenta'
fluidsynth_dir = 'fluidsynth'
chroma_dir = 'NNLS_Chroma'

magenta_config = 'attention_rnn'
gen_music_len = 2048
gen_music_num = 2
#start_seed = [60, -2, 60, -2, 67, -2, 67, -2]

soundfont_file = 'GS.sf2'

NNLS_config = 'chroma_spec_1.n3'



# In[62]:


for i in range(gen_music_num):
    file = open("Magenta_bash.sh", "w")
    file.write("source activate magenta\n")
    file.write("melody_rnn_generate --config=\'"+magenta_config+"\'"+               " --bundle_file=\""+op.join(working_dir, magenta_config+".mag")+"\""+               " --output_dir=\""+op.join(working_dir, magenta_dir)+"\""+               " --num_outputs=1"+
               " --num_steps="+str(gen_music_len)+
               " --primer_melody=\"[60]\"\n")
    file.write("source deactivate magenta")
    file.close()
    subprocess.call("chmod 755 Magenta_bash.sh", shell=True)
    subprocess.call("./Magenta_bash.sh", shell=True)


# In[65]:


k = 0
for f in listdir(op.join(working_dir, magenta_dir)):
    file = open("fluid_config.txt", "w")
    file.write("load "+ "GS.sf2\n")
    if k%3==0:
        file.write("select 0 1 0 0\n") #11/39 128/1
    elif k%3==1:
        file.write("select 0 1 11 39\n")
    else:
        file.write("select 0 1 128 1\n")
    file.close()
    subprocess.call("fluidsynth -i -f fluid_config.txt -F "+op.join(working_dir, fluidsynth_dir, "gen_music_"+str(k)+'.wav')+" -o player.reset-synth=False "+op.join(magenta_dir, f), shell=True)
    k=k+1


# In[74]:


file = open("NNLS_bash.sh", "w")
file.write("source ~/.bashrc\n")
file.write("sonic-annotator -t "+op.join(working_dir, NNLS_config)+" -r "+op.join(working_dir, fluidsynth_dir)+" -w csv --csv-force --csv-basedir "+op.join(working_dir, chroma_dir))
file.close()

subprocess.call("chmod 755 NNLS_bash.sh", shell=True)
subprocess.call("./NNLS_bash.sh", shell=True)

