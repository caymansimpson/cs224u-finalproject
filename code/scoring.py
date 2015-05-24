def score_model(guess_answer_pairs):
    unscaled_score = 0.0
    for guess, answer in guess_answer_pairs:
        if guess == answer:
            unscaled_score += 1.0
        elif guess == -1:
            unscaled_score += 0.0
        else:
            unscaled_score -= 0.25
    return scale_score(unscaled_score)

def scale_score(unscaled_score):
    score_conversion_table = dict([(67, 800), (31, 500), (66, 800), (30, 500), 
                                   (65, 800), (29, 490), (64, 790), (28, 480), 
                                   (63, 770), (27, 480), (62, 760), (26, 470),
                                   (61, 740), (25, 460), (60, 730), (24, 460), 
                                   (59, 720), (23, 450), (58, 700), (22, 440), 
                                   (57, 690), (21, 440), (56, 680), (20, 430),
                                   (55, 670), (19, 420), (54, 670), (18, 410), 
                                   (53, 660), (17, 410), (52, 650), (16, 400), 
                                   (51, 640), (15, 390), (50, 630), (14, 380),
                                   (49, 620), (13, 380), (48, 620), (12, 370), 
                                   (47, 610), (11, 360), (46, 600), (10, 350), 
                                   (45, 600), (9, 340), (44, 590), (8, 330),
                                   (43, 580), (7, 320), (42, 570), (6, 310), 
                                   (41, 570), (5, 300), (40, 560), (4, 290), 
                                   (39, 550), (3, 270), (38, 550), (2, 260),
                                   (37, 540), (1, 240), (36, 530), (0, 220), 
                                   (35, 530), (-1, 210), (34, 520), (-2, 200)])
    rounded_unscaled_score = int(round(unscaled_score))
    return score_conversion_table[rounded_unscaled_score]


# test_pairs = [(0,0), (1,0), (2,2), (1,1), (-1,1)]
# print "Score: " + str(score_model(test_pairs))