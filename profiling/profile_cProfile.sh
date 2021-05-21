now=`date +"%m-%d-%Y %H:%M:%S"`
filename="profiling/profiling_cProfile_${now}.log"
pypy -m cProfile -s cumulative src/coeus.py --profile "r5k1/1p4pp/1p6/2pp2P1/4rb2/P1P2N2/1P1B3P/4RK2 w - - 0 27" > "${filename}"
echo "Profile saved to ${filename}"