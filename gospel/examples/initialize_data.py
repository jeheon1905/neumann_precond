def initialize(data_dir = "./data/",  tgz_filename ="pz.0.3.1.tgz"):
    import os 
    import urllib.request
    import tarfile
    os.makedirs(data_dir, exist_ok=True)
    if( not os.path.exists(tgz_filename) ):
        print(f"data directory({data_dir}) does not exist")
        urllib.request.urlretrieve(f"https://theos-wiki.epfl.ch/public/Main/NoBackup/{tgz_filename}", data_dir+"/"+tgz_filename )

    with tarfile.open(data_dir+"/"+tgz_filename, 'r:gz') as f:
        f.extractall(data_dir)

if __name__=="__main__":
    initialize('data_lda', "pz.0.3.1.tgz")        
    initialize('data_gga', 'pbe.0.3.1.tgz')        
