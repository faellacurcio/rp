samples = [[0.44208079625387986, 0.2209381799898023, 0.05434345708758807, -0.04926351502623683, 0.07190884098243902, 0.07605376969829411] ;
    [0.44208079625387986, -0.09811522076364042, 0.05434345708758807, 0.07786394158843211, 0.1165140499120439, -0.06071069574434759];
    [0.44208079625387986, -0.1090292786607704, 0.05434345708758807, 0.12564979328026415, 0.10549696221193716, 0.09539015971190154];
    [0.44208079625387986, -0.1134725778530741, -0.0029299685645841683, 0.17841058902057194, 0.09835540391204009, -0.01061765756074874];
    [0.44208079625387986, 0.2209381799898023, 0.02240281886744182, 0.09389279867531639, 0.07802492443067725, 0.08132457455327946];
    [0.44208079625387986, 0.22635640148774866, 0.20910403700328503, 0.08755773422694098, 0.16861753874698787, 0.13404433790711553];
    [0.44208079625387986, 0.22635640148774866, 0.20777718379518167, 0.08755773422694098, 0.056483012031001284, 0.074596214982262];
    [0.44208079625387986, 0.22635640148774866, 0.20910403700328503, 0.008210716026638087, 0.10042499526535954, 0.0686373271360948]];
disp(samples)
plot([2,3,4,5,6,7],mean(samples,1))
title('M�dia das silhuetas')