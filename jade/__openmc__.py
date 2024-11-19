# Handle OpenMC optional dependency with global flag. 
try:
    import openmc
    OMC_AVAIL = True
except ImportError:
    OMC_AVAIL = False
    print("OpenMC has not been installed - see JADE installation instructions")