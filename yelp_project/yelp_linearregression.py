import graphlab as gl

#simple regression model to predict star ratings of the restuarant 
#using user & business average stars & users and business review count.

#Read the review, business and user yelp data into graphlab Sframe for linear regression.

review = gl.Sframe('review.csv')
user = gl.Sframe('user.csv')
business = gl.Sframe('business.csv')


#join business, review and user dataset 

join_review_business = business.join(review, how='inner', on='business_id')

join_review_business_user = join_review_business.join(user, how = 'inner', on = "user_id") 


# Do a random split on the joined dataset into training and test sets.

train_set, test_set = user_business_review_table.random_split(0.7, seed=5)

#Train the model using the training sets

model = gl.linear_regression.create(train_set, target='stars', 
                                    features = ['business_stars','business_review_count', 
                                                'user_review_count', 'user_average_stars'])

#predict the linears regression result

eval= model.predict(test_set)

print(eval)

#finds the mean square error.

result = model.evaluate(test_set)

print(result)
