# https://matplotlib.org/users/customizing.html

#figure.titlesize : small        # size of the figure title (Figure.suptitle())
figure.figsize   : 8, 5         # figure size in inches
#figure.dpi       : 200          # figure dots per inch
figure.facecolor : white
figure.edgecolor : white
figure.autolayout : True        # Automatically apply 'plt.tight_layout'

lines.linewidth : 3
lines.markersize : 10
legend.fontsize : 15
legend.handlelength : 1
axes.titlesize : 17
axes.labelsize : 15
axes.linewidth : 1
axes.grid : True
grid.linestyle : :
xtick.labelsize : 14 # font size of tick labels
xtick.direction : in
ytick.direction : in
ytick.labelsize : 14
xtick.major.size : 10      # major tick size in points
xtick.minor.size : 5     # minor tick size in points
ytick.major.size : 10
ytick.minor.size : 5

#xtick.major.size : 20
#xtick.major.width : 4
#xtick.minor.size : 10
#xtick.minor.width : 2

xtick.minor.visible  : True   # turn minor ticks on by default
ytick.minor.visible  : True

# --- I like this but most people don't have it
#font.family : Optima

# --- this one is still distinct but should be more widely usable
#font.family : Courier New
#font.weight : bold
#axes.labelweight : bold

# -- this font is close to phys rev
font.family : STIXGeneral
font.serif : STIX
mathtext.fontset : stix
