import os
import pydub

test_files = os.listdir("test_ref/IndianClassical")

for file in test_files : 
    audio = pydub.AudioSegment.from_file(os.path.join("test_ref/IndianClassical",file))
    
    audio = audio[:10000]
    audio.export("test_ref/"+file, format="wav")
    
