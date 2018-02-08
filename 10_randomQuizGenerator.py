import random
import os

capitals = {'Alabama': 'Montgomery', 'Alaska': 'Juneau', 'Arizona': 'Phoenix',
            'Arkansas': 'Little Rock', 'California': 'Sacramento', 'Colorado': 'Denver',
            'Connecticut': 'Hartford', 'Delaware': 'Dover', 'Florida': 'Tallahassee',
            'Georgia': 'Atlanta', 'Hawaii': 'Honolulu', 'Idaho': 'Boise', 'Illinois':
                'Springfield', 'Indiana': 'Indianapolis', 'Iowa': 'Des Moines', 'Kansas':
                'Topeka', 'Kentucky': 'Frankfort', 'Louisiana': 'Baton Rouge', 'Maine':
                'Augusta', 'Maryland': 'Annapolis', 'Massachusetts': 'Boston', 'Michigan':
                'Lansing', 'Minnesota': 'Saint Paul', 'Mississippi': 'Jackson', 'Missouri':
                'Jefferson City', 'Montana': 'Helena', 'Nebraska': 'Lincoln', 'Nevada':
                'Carson City', 'New Hampshire': 'Concord', 'New Jersey': 'Trenton',
            'New Mexico': 'Santa Fe', 'New York': 'Albany', 'North Carolina': 'Raleigh',
            'North Dakota': 'Bismarck', 'Ohio': 'Columbus', 'Oklahoma': 'Oklahoma City',
            'Oregon': 'Salem', 'Pennsylvania': 'Harrisburg', 'Rhode Island': 'Providence',
            'South Carolina': 'Columbia', 'South Dakota': 'Pierre', 'Tennessee':
                'Nashville', 'Texas': 'Austin', 'Utah': 'Salt Lake City', 'Vermont':
                'Montpelier', 'Virginia': 'Richmond', 'Washington': 'Olympia',
            'WestVirginia': 'Charleston', 'Wisconsin': 'Madison', 'Wyoming': 'Cheyenne'}

# 30份试卷
for quizNum in range(30):
    # Create the quiz and answer key files
    quizFile = open('./quiz/quiz{0}.txt'.format(quizNum + 1), 'w')
    answerFile = open('./quiz/answer{0}.txt'.format(quizNum + 1), 'w')

    # Write out the header for the quiz.
    quizFile.write('name:\ndate:\n\n')
    quizFile.write((' ' * 20) + 'State Capitals Quiz (Form %s)' % (quizNum + 1))
    quizFile.write('\n\n')

    # Shuffle the order of the states.
    states = list(capitals.keys())
    random.shuffle(states)

    # Loop through all 50 states, making a question for each.
    for questionNum in range(len(capitals)):
        quizFile.write('{0}. What is the capital of {1}\n'.format(questionNum + 1, states[questionNum]))

        correctAnswer = capitals[states[questionNum]]
        wrongAnswers = list(capitals.values())
        wrongAnswers = random.sample(wrongAnswers, 3)
        answers = [correctAnswer] + wrongAnswers
        random.shuffle(answers)
        for i in range(4):
            quizFile.write('{0}. {1}\n'.format('ABCD'[i], answers[i]))

        quizFile.write('\n')

        # write the answers
        answerFile.write('{0}. {1}'.format(questionNum + 1, 'ABCD'[answers.index(correctAnswer)]))
        answerFile.write('\n')

    quizFile.close()
    answerFile.close()
