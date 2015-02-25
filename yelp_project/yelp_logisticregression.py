import graphlab as gl

#Logistic Regression Model to predict whether a resturant is good or bad.

# If the average star rating is greater than or equal to 3 , the restuarant is predicted as good else bad.

#Read the review, business and user yelp data into graphlab Sframe for linear regression.

review = gl.Sframe('review.csv')
user = gl.Sframe('user.csv')
business = gl.Sframe('business.csv')


#join business, review and user dataset 

join_review_business = business.join(review, how='inner', on='business_id')

join_review_business_user = join_review_business.join(user, how = 'inner', on = "user_id") 


# Predict class label. If the average star rating is greater than or equal to 3 , the restuarant is predicted as good else bad.

join_review_business['good'] = join_review_business['stars'] >= 3

train_set, test_set = user_business_review_table.random_split(0.7, seed=5)

#Train the model using the training sets

model = gl.logistic_classifier.create(train_set, target='good', 
                                    features = ['business_stars','business_review_count', 
                                                'user_review_count', 'user_average_stars'])

#predict the logistic classifier

model= model.predict(test_set)


#finds the mean square error.

result = model.evaluate(test_set)

#print results

print(result['accuracy'])
print(result['confusion_matrix'])
