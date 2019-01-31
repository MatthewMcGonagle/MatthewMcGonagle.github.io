excludes="--exclude=*.svg --exclude=*.dat --exclude=*.exe --exclude=*.obj --exclude=*.swp --exclude=*.exp --exclude=*.lib"
if [ "$1" == "" ] 
then
    echo "Missing target"

elif [ "$2" == "" ]
then
    echo "Missing source"

else
    echo "Making tarball for $1"
    echo "       source = $2"
    echo tar czvf $1 $2 $excludes
    tar czvf $1 $2 $excludes
fi
