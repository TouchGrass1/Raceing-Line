# Raceing-Line
A software to find the racing line of a race track

For simplification, I will use Indy cars

Variable I would like to include: tyre type(soft,medium, hard, slick), tyre condition(dynamically deteriate as time increases), surface temp, tyre temp(decrease during straights, increase during turns)
What it may affect - friction, therefore turning radius before slip, max accel/ deccel

Plan:
Phase 1 - Draw track (turtle)
How - convert a black and white image of a track into 2 sections: a pixel array for the inside of the track, a pixel array for the outside of the track.
Phase 2 - Divide the track into straights and turns
How - divide the track into 10cm sections, then iterate through each 1cm block, find the gradient of that block, if it is roughly the same for all blocks, then it is a straight, else a turn.
      to find the center and radius, draw a normal from the tangents, and the point where they intersect is the center. Use pythag to find r
Phase 3 - find the maximum speed the vehicle can take during a turn (will vary with how wide the vehicle goes, start with Apex)
How - using circular motion and combining other variables into a formula
Phase 4 - find the minimum distance to complete the track
Phase 5 - find max speed on straight
How - back track from turns
Phase 6 - vary for different racing lines and then find the optimum
Phase 7 - draw race line, colour graded for speed

          
