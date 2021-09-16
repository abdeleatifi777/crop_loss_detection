import os
import subprocess

#datadir = '/u/58/wittkes3/unix/Documents/Landsat/espa-bulk-downloader-master/data/espa-samantha.wittke@nls.fi-0101906059812'
#datadirs = ['/scratch/cs/ai_croppro/data/landsat/2013', '/scratch/cs/ai_croppro/data/landsat/2014', '/scratch/cs/ai_croppro/data/landsat/2015']
#datadirs = ['/scratch/cs/ai_croppro/data/ls7/espa-samantha.wittke@nls.fi-11252019-083854-732','/scratch/cs/ai_croppro/data/ls7/espa-samantha.wittke@nls.fi-11252019-083915-160','/scratch/cs/ai_croppro/data/ls7/espa-samantha.wittke@nls.fi-11252019-083935-391','/scratch/cs/ai_croppro/data/ls7/espa-samantha.wittke@nls.fi-11252019-083947-932','/scratch/cs/ai_croppro/data/ls7/espa-samantha.wittke@nls.fi-11252019-084521-622']
datadirs= ['/scratch/cs/ai_croppro/data/ls7/espa-samantha.wittke@nls.fi-11252019-084521-622']

for datadir in datadirs:
    for tar in os.listdir(datadir):
        if tar.endswith('.tar.gz'):
            gz = os.path.splitext(tar)
            print(gz)
            name = os.path.splitext(gz[0])
            path = os.path.join(datadir,tar)
            print(path)
            newpath = os.path.join(datadir,name[0])
            if not os.path.exists(newpath):
                os.mkdir(newpath)
                print(newpath)
                subprocess.call('tar -xvzf '+ path + ' -C '+newpath, shell=True)
