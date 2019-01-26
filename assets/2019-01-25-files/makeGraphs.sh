if [ "$1" == "" ]
then
    dimX=20
    dimY=20
else
    dimX=$1
    dimY=$1
fi
 
gnuplot -e "filename = 'data/values.dat'" -e "outname = 'Values'" -e "dimX = '$dimX'" -e "dimY = '$dimY'" fromFile.p
gnuplot -e "filename = 'data/errors.dat'" -e "outname = 'Errors'" -e "dimX = '$dimX'" -e "dimY = '$dimY'" fromFile.p
gnuplot -e "filename = 'data/log10RelErrors.dat'" -e "outname = 'Log10RelErrors'" -e "dimX = '$dimX'" -e "dimY = '$dimY'" fromFile.p
