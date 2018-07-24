from libs.SongProcessing import Song

# Declarations
forTraining, forTesting = [Song('./data/datasets/forTraining.txt'), Song('./data/datasets/forTesting.txt')]

print('Generating datasets...')

for ech_numb in range(4):

    # Class 1 training data generation
    forTraining.saveSong(forTraining.getSongSignature('./data/training/Samba/samba_music_ech_' \
         + str(ech_numb) + '.wav'), 1)
    
    # Class 0 training data generation
    forTraining.saveSong(forTraining.getSongSignature('./data/training/HipHop/hiphop_music_ech_' \
        + str(ech_numb) + '.wav'), 0)

# Test data generation for prediction
forTesting.saveSong(forTesting.getSongSignature('./data/testing/hiphop.wav'), None)
forTesting.saveSong(forTesting.getSongSignature('./data/testing/samba.wav'), None)

print('Done !')