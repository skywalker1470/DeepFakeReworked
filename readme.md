run the commands to build a model yourself:
(On linux) idk windows commands to create a venv so sorry,  
---> python3 -m venv venv (or) python -m venv venv   
---> source venv/bin/activate    
---> pip install -r requirements.txt  
---> python data_create.py   
---> python extract_frames.py  
---> python prepare_celebdf_list.py   
---> python train_celebdf_xception (We changed the model to EfficientNet for better results, naming scheme wrong)  
---> python testing.py  

(Now we run the app for your own videos)
--->python app.py  
--->Go to http://127.0.0.1:5000 for your app, ENJOY!!
