set terminal svg size 400,200 dynamic

set title outname." Heat Plot"
set output 'graphs/'.outname.'_heat_'.dimX.'_'.dimY.'.svg'
plot filename binary record=(dimX,dimY) scan=yx with image notitle 

set title outname." Surface Plot"
set grid
set hidden3d
set output 'graphs/'.outname.'_surface_'.dimX.'_'.dimY.'.svg'
splot filename binary record=(dimX,dimY) scan=yx with lines notitle 
