import cv2
import numpy as numpy
import os,sys,time


def face_detection(image):
    size = 4
    haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    (im_width,im_height) = (68,68)
    """
    haar_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mini = cv2.resize(gray, ((int)(gray.shape[1] / size), (int)(gray.shape[0] / size)))
    face = haar_classifier.detectMultiScale(mini)
    (x,y,w,h) = face[0]
    """
    im = cv2.flip(image,1,0)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    mini = cv2.resize(gray, ((int)(gray.shape[1] / size), (int)(gray.shape[0] / size)))
    faces = haar_cascade.detectMultiScale(mini)
    #faces = sorted(faces, key=lambda x: x[3])
    try:
        if faces.any():
            face_i = faces[0]
            (x,y,w,h) = [v * size for v in face_i]
            face = gray[y:y+h, x:x+w]
            face_resize_final = cv2.resize(face, (im_width, im_height))
            cv2.rectangle(im, (x,y), (x+w,y+h), (0,255,0),3)
        else:
            print("Could not detect face please try again...")
            return None,None,None
    except:
        print("Could not detect face please try again...")
        return None,None,None


    return face_resize_final, (x,y,w,h), im


def predict_image(test_image,model,names):
    img = test_image.copy()
    face, bounding_box, img = face_detection(img)

    if (face == None or bounding_box == None or img == None):
        return None

    (x,y,w,h) = bounding_box
    label = model.predict(face)
    #label_text = database[label-1]
    name = names[label[0]]

    #print (label_text)
    cv2.putText(img,name+', Confidence(%): '+("%.2f" % label[1]),(x-10,y-10),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0))

    return img

model = cv2.face.LBPHFaceRecognizer_create()
while True:

    user=int(input('1.Register a face 2.Recognize a face 3.Quit: '))

    if user == 1:
        count = 0
        size = 4
        fn_harr = 'haarcascade_frontalface_default.xml'
        fn_dir = 'database'
        fn_name = input('Enter name of person: ')
        path = os.path.join(fn_dir,fn_name)

        if not os.path.isdir(path):
            os.mkdir(path)

        (im_width,im_height) = (68,68)
        haar_cascade = cv2.CascadeClassifier(fn_harr)
        webcam = cv2.VideoCapture(0)

        print('----------Taking pictures----------')
        print('--------Give some expressions--------')

        while count<64:
            (rval,im) = webcam.read()
            print("Image " + str(count+1) + "/64 taken...")
            im = cv2.flip(im,1,0)
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            mini = cv2.resize(gray, ((int)(gray.shape[1] / size), (int)(gray.shape[0] / size)))
            faces = haar_cascade.detectMultiScale(mini)
            faces = sorted(faces, key=lambda x: x[3])
            if faces:
                face_i = faces[0]
                (x,y,w,h) = [v * size for v in face_i]
                face = gray[y:y+h, x:x+w]
                face_resize = cv2.resize(face, (im_width, im_height))
                pin = sorted([int(n[:n.find('.')]) for n in os.listdir(path) if n[0]!='.']+[0])[-1]+1
                cv2.imwrite('%s/%s.png' % (path,pin), face_resize)
                cv2.rectangle(im, (x,y), (x+w,y+h), (0,255,0),3)
                cv2.putText(im,fn_name,(x-10,y-10),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0))
                time.sleep(0.38)
                count += 1
            
            cv2.imshow('OpenCV', im)
            key = cv2.waitKey(10)
            if key == 27:
                break
        print("Images taken and saved to " + fn_name + " folder in database")
        cv2.destroyAllWindows()
        webcam.release()

    elif user == 2:
        fn_dir = 'database'

        print('\nTraining...')
    
        (images,labels,names,id) = ([],[],{},0)
        for(subdirs,dirs,files) in os.walk(fn_dir):
            for subdir in dirs:
                names[id] = subdir
                subjectpath = os.path.join(fn_dir,subdir)
                for filename in os.listdir(subjectpath):
                    path = subjectpath+'/'+filename
                    label = id
                    images.append(cv2.imread(path,0))
                    labels.append(int(label))
                id += 1
        (images,labels) = [numpy.array(lis) for lis in [images, labels]]
        
        #facesF, labelsF = prepare_data(path)

        model.train(images, labels)

        print('Done')

        print("\nWe are going to take a sample picture for verification.")
        print("Please look at the camera")
        time.sleep(4)
        cap = cv2.VideoCapture(0) # video capture source camera (Here webcam of laptop) 
        ret,frame = cap.read() # return a single frame in variable `frame`
        print("Press key 'y' to save the sample")
        while(True):
            cv2.imshow('img1',frame) #display the captured image
            if cv2.waitKey(1) & 0xFF == ord('y'): #save on pressing 'y' 
                test_image = 'temp.png'
                cv2.imwrite(test_image,frame)
                cv2.destroyAllWindows()
                break

        cap.release()

        predict_test = cv2.imread(test_image)
        predict = predict_image(predict_test,model,names)

        if predict == None:
            continue

        cv2.imshow('Face Recognition', predict)
        print("Press any key to continue...\n")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        os.remove(test_image)
    else:
        sys.exit()

        

        
            