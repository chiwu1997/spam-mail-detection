import json
import boto3
import os
from botocore.exceptions import ClientError
# from sms_spam_classifier_utilities import one_hot_encode
# from sms_spam_classifier_utilities import vectorize_sequences
import string
import sys
from hashlib import md5

if sys.version_info < (3,):
    maketrans = string.maketrans
else:
    maketrans = str.maketrans

vocabulary_length = 9013
RECEIVER = 'cloud@ww6998.com'
RECEIVER = os.environ.get("spam_detect_email_addr")
end_point = 'sms-spam-classifier-mxnet-2021-11-18-22-20-40-026'
end_point = os.environ.get("sagemaker_endpoint")
client = boto3.client('sagemaker-runtime')

fail_response = {
    'statusCode': 501,
    'body': json.dumps('Not Implemented')
}

okay_response = {
    'statusCode': 200,
    'body': json.dumps('Spam analysis result sent')
}

def lambda_handler(event, context):
    print(event)
    # bucket: email-files-storing-bucket
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']
    print(bucket, key)
    
    # parse email subject & content
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket, Key=key)
    if response['ResponseMetadata']['HTTPStatusCode'] == 200:
        email_object = response['Body'].read().decode('utf-8')
        email = parser(email_object)
        print('#email body: ', email['body'])
        
        # sagemaker analysis
        one_hot_test_messages = one_hot_encode([email['body']], vocabulary_length)
        encoded_test_messages = vectorize_sequences(one_hot_test_messages, vocabulary_length)
        # payload = json.dumps(encoded_test_messages.tolist())
        payload = json.dumps(encoded_test_messages)
        res_sagemaker = client.invoke_endpoint(EndpointName=end_point, ContentType='application/json', Body=payload)
        print("#res_sagemaker: ", res_sagemaker)
        
        # resend email
        if res_sagemaker['ResponseMetadata']['HTTPStatusCode'] == 200:
            result = json.loads(res_sagemaker['Body'].read().decode())
            print(email, result)
            email_handler(email, result)
            return okay_response
        else:
            return fail_response
    else:
        return fail_response


def parser(payload):
    email = dict()
    rows = payload.split('\r\n')
    for i, row in enumerate(rows):
        if 'sender' not in email:
            if 'From: ' in row:
                email['sender'] = row[6:]
                print("#sender: ", email['sender'])
        if 'Subject: ' in row:
            email['subject'] = row[9:]
            print("#subject: ", email['subject'])
            
        if 'X-SES-Outgoing' in row:
            email['date'] = row[16:26]
        
        if 'Content-Transfer-Encoding: 7bit' in row:
            email['body'] = rows[i+2]
            break
    
    print('#body: ', email['body'])
    # email['received'] = payload.split('Received: ')[1].split('for ' + RECEIVER + ';\r\n ')[1].split('\r\n')[0]
    # email['body'] = payload.split('Content-Type: text/plain; charset="UTF-8"\r\n\r\n')[1].split('\r\n\r\n')[0].replace('\n', '')

    return email
  
  
def email_handler(email, result):
    # Replace sender@example.com with your "From" address.
    # This address must be verified with Amazon SES.
    classification = 'SPAM' if result['predicted_label'][0][0] == 1 else 'HAM'
    # If necessary, replace us-west-2 with the AWS Region you're using for Amazon SES.
    AWS_REGION = "us-east-1"

    # The subject line for the email.
    SUBJECT = "Spam filter notification"

    confidence = result['predicted_probability'][0][0]
    if classification == 'HAM':
        confidence = 1 - result['predicted_probability'][0][0]

    # The email body for recipients with non-HTML email clients.
    BODY_TEXT = 'We received your email sent at ' + email['date'] + \
                ' with the subject [' + email['subject'] + '].\n\n' + \
                'Here is a 240 character sample of the email body:\n' + \
                email['body'][:240] + \
                '\n\nThe email was categorized as ' + classification + \
                ' with a ' + str(confidence * 100) + '% confidence.”'


    # The HTML body of the email.
    BODY_HTML = BODY_TEXT

    # The character encoding for the email.
    CHARSET = "UTF-8"

    # Create a new SES resource and specify a region.
    client = boto3.client('ses', region_name=AWS_REGION)

    # Try to send the email.
    try:
        # Provide the contents of the email.
        response = client.send_email(
            Destination={
                'ToAddresses': [
                    email['sender'],
                ],
            },
            Message={
                'Body': {
                    'Html': {
                        'Charset': CHARSET,
                        'Data': BODY_HTML,
                    },
                    'Text': {
                        'Charset': CHARSET,
                        'Data': BODY_TEXT,
                    },
                },
                'Subject': {
                    'Charset': CHARSET,
                    'Data': SUBJECT,
                },
            },
            Source=RECEIVER
            # If you are not using a configuration set, comment or delete the
            # following line
            # ConfigurationSetName=CONFIGURATION_SET,
        )
        print(response)
    # Display an error if something goes wrong.
    except ClientError as e:
        print("testtttttttt#######")
        print(e.response['Error']['Message'])
    # else:
    print("Email sent! Message ID:"),
    # print(response['MessageId'])

def vectorize_sequences(sequences, vocabulary_length):
    # results = np.zeros((len(sequences), vocabulary_length))
    results = [[0 for _ in range(vocabulary_length)] for _ in range(len(sequences))]
    for i, sequence in enumerate(sequences):
        # results[i, sequence] = 1.
        for j in sequence:
            results[i][j] = 1.
    return results


def one_hot_encode(messages, vocabulary_length):
    data = []
    for msg in messages:
        temp = one_hot(msg, vocabulary_length)
        data.append(temp)
    return data


def text_to_word_sequence(text,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True, split=" "):
    """Converts a text to a sequence of words (or tokens).
    # Arguments
        text: Input text (string).
        filters: list (or concatenation) of characters to filter out, such as
            punctuation. Default: `!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n`,
            includes basic punctuation, tabs, and newlines.
        lower: boolean. Whether to convert the input to lowercase.
        split: str. Separator for word splitting.
    # Returns
        A list of words (or tokens).
    """
    if lower:
        text = text.lower()

    if sys.version_info < (3,):
        if isinstance(text, unicode):
            translate_map = dict((ord(c), unicode(split)) for c in filters)
            text = text.translate(translate_map)
        elif len(split) == 1:
            translate_map = maketrans(filters, split * len(filters))
            text = text.translate(translate_map)
        else:
            for c in filters:
                text = text.replace(c, split)
    else:
        translate_dict = dict((c, split) for c in filters)
        translate_map = maketrans(translate_dict)
        text = text.translate(translate_map)

    seq = text.split(split)
    return [i for i in seq if i]


def one_hot(text, n,
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
            lower=True,
            split=' '):
    """One-hot encodes a text into a list of word indexes of size n.
    This is a wrapper to the `hashing_trick` function using `hash` as the
    hashing function; unicity of word to index mapping non-guaranteed.
    # Arguments
        text: Input text (string).
        n: int. Size of vocabulary.
        filters: list (or concatenation) of characters to filter out, such as
            punctuation. Default: `!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n`,
            includes basic punctuation, tabs, and newlines.
        lower: boolean. Whether to set the text to lowercase.
        split: str. Separator for word splitting.
    # Returns
        List of integers in [1, n]. Each integer encodes a word
        (unicity non-guaranteed).
    """
    return hashing_trick(text, n,
                         hash_function='md5',
                         filters=filters,
                         lower=lower,
                         split=split)


def hashing_trick(text, n,
                  hash_function=None,
                  filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                  lower=True,
                  split=' '):
    """Converts a text to a sequence of indexes in a fixed-size hashing space.
    # Arguments
        text: Input text (string).
        n: Dimension of the hashing space.
        hash_function: defaults to python `hash` function, can be 'md5' or
            any function that takes in input a string and returns a int.
            Note that 'hash' is not a stable hashing function, so
            it is not consistent across different runs, while 'md5'
            is a stable hashing function.
        filters: list (or concatenation) of characters to filter out, such as
            punctuation. Default: `!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n`,
            includes basic punctuation, tabs, and newlines.
        lower: boolean. Whether to set the text to lowercase.
        split: str. Separator for word splitting.
    # Returns
        A list of integer word indices (unicity non-guaranteed).
    `0` is a reserved index that won't be assigned to any word.
    Two or more words may be assigned to the same index, due to possible
    collisions by the hashing function.
    The [probability](
        https://en.wikipedia.org/wiki/Birthday_problem#Probability_table)
    of a collision is in relation to the dimension of the hashing space and
    the number of distinct objects.
    """
    if hash_function is None:
        hash_function = hash
    elif hash_function == 'md5':
        hash_function = lambda w: int(md5(w.encode()).hexdigest(), 16)

    seq = text_to_word_sequence(text,
                                filters=filters,
                                lower=lower,
                                split=split)
    return [int(hash_function(w) % (n - 1) + 1) for w in seq]