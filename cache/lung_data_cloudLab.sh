#!/bin/bash
#cd /
curl -L -o my_data.tar.gz https://www.dropbox.com/scl/fi/t5c09f1mam7ljawwnr7d0/lung_4_sub.tar.gz?rlkey=0ujafdjnptg3ja31d1iq0x6yq&dl=0
tar -zxvf my_data.tar.gz
mv lung_80_subFolders data
#find name '._*' delete
cd data
find . -name '._*' -delete
cd test/lung_aca/aca_1                                                                                                                                                                   
mv *jpeg /users/aga5h3/data/test/lung_aca                                                                                                                                               
cd ..                                                                                                                                                                      
rm -r aca*                                                                                                                                                             
cd ..                                                                                                                                                                      
cd lung_n                                                                                                                                                                
cd n_1                                                                                                                                                               
mv *jpeg /users/aga5h3/data/test/lung_n                                                                                                                                     
cd ..                                                                                                                                                                      
rm -r n_*                                                                                                                                                                
cd .. 
cd lung_scc 
cd scc_1 
mv *jpeg /users/aga5h3/data/test/lung_scc      
cd ..     
rm -r scc* 
cd ../.. 
cd train                                                                                                                                                                                                                                                                                                                                           
cd lung_aca                                                                                                                                                               
cd aca_1                                                                                                                                                               
mv *jpeg /users/aga5h3/data/train/lung_aca                                                                                                                                               
cd ..                                                                                                                                                                      
rm -r aca*                                                                                                                                                               
cd ..                                                                                                                                                                      
cd lung_n                                                                                                                                                                
cd n_1                                                                                                                                                               
mv *jpeg /users/aga5h3/data/train/lung_n                                                                                                                                             
cd ..                                                                                                                                                                      
rm -r n_*                                                                                                                                                               
cd ..                                                                                                                                                                      
cd lung_scc                                                                                                                                                               
cd scc_1                                                                                                                                                            
mv *jpeg /users/aga5h3/data/train/lung_scc                                                                                                                                           
cd ..                                                                                                                                                                      
rm -r scc*
cd ../../.. 
echo "Done!!"
