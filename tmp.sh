for file in $(ls ./*.hip); do mv $file ${file::-4}; done
