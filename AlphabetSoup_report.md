# Alphabet Soup Challenge Report


## Overview of the Analysis

The purpose of the analysis was to create a deep learning model to show enable prediction of the success for charitable organizations. Multiple features were used for this challenge, amoung them were classification, organization type, affiliation and others. The model attempts to identify contributing factors the would lead to the success of the orginazations pursuits.

Data Preprocessing

- What variable(s) are the target(s) for your model?

    The Target value used for both models was "IS_SUCCESFUL", This shows us if the organization was successful for their fundrasing goals.

- What variable(s) are the features for your model?

    Feature values from the original data were APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, and ASK_AMT.


- What variable(s) should be removed from the input data because they are neither targets nor features?

     The columns 'EIN','NAME' were dropped from the input data as the were not neccessary.

     For the optimized data the columns SPECIAL_CONSIDERATIONS, STATUS, and ASK_AMT were also removedv.
     




## Compiling, Training, and Evaluating the Model

- How many neurons, layers, and activation functions did you select for your neural network model, and why?

    Three (3) layers were used, layer1 used 10 neurons, layer2 used 8 neurons and layer3 used 6 neurons. Activations were "relu" and "sigmoid". The layers were chosen randomly unit we were able to find the most effective to achieve the greatest efficiency, although we were not able to achieve an accuracy of 75% or greater. 


- Were you able to achieve the target model performance?

    We were only able to achive and accuracy of 73.14% (0.7314)

- What steps did you take in your attempts to increase model performance?

    We changed multiple values the number of layers, nodes, acitvations and the number of epochs to try to achieve 75% accuracy but were not able to achieve the goal of 75% accuracy.


## Summary: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation

The original and optimized models presented reasonable accuracy levels for predictions of successful campaigns. Perhaps adjusting architecture, activations and the number of epochs variables could lead to greater performance of these models.
