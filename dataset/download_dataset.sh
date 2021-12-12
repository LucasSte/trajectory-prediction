# Training set #1
wget https://mirror.robojackets.org/robocup-small-size-league/gamelogs/2019/div-a/2019-07-05_06-15_RoboTeam_Twente-vs-RoboDragons.log.gz

gzip -dk 2019-07-05_06-15_RoboTeam_Twente-vs-RoboDragons.log.gz
mv 2019-07-05_06-15_RoboTeam_Twente-vs-RoboDragons.log data_set_1.log

# Training set #2
wget https://mirror.robojackets.org/robocup-small-size-league/gamelogs/2019/div-b/2019-07-06_06-18_RoboJackets-vs-nAMeC.log.gz

gzip -dk 2019-07-06_06-18_RoboJackets-vs-nAMeC.log.gz
mv 2019-07-06_06-18_RoboJackets-vs-nAMeC.log.gz data_set_2.log

# Testing set
wget https://mirror.robojackets.org/robocup-small-size-league/gamelogs/2019/div-a/2019-07-07_05-26_ER-Force-vs-MRL.log.gz

gzip -dk 2019-07-07_05-26_ER-Force-vs-MRL.log.gz
mv 2019-07-07_05-26_ER-Force-vs-MRL.log.gz data_set_3.log

