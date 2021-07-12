# <center>User Manual for Learn-Forget-Memorize</center>
### Table of Contents
<ol>
    <li> How it works </li>
    <li> Main Screen </li>
    <li> File</li>
    <li> Settings </li>
    <li> Progress </li>
    <li> Machine Learning </li>
    <li> Data </li>
    <li> Edit Screen </li>
    <li> Adding Flashcards </li>
    <li> Taking Tests </li>
</ol>        

------------------------------------

## 1. How it Works
<p> Learn-Forget-Memorize is a study application that is made to make flashcards quickly, organize them efficiently and keep track of when you might forget the material.

</p> <p>There are 3 stages in the development of studying material
    <br> <b> Learning </b> -in this stage you are simply working your way up to an acceptable amount of knowledge of the given material.
    <br><b> Forgetting </b> -in this stage you are slowly (or quickly) forgetting the material you learned  
    <br><b> Memorized</b>  - in this stage the information you learned is permenant or will take so long to forget that it might as well be considered memorized</p> 
    
<p> This application works by allowing the user to section the material into the three categories. Flashcards are made by the user, the material is studied and then tests are taken on the material. If you feel like you have achieved a high enough score on the material then you can move the material into the forgetting category where the application will guess how quickly you are forgetting the material (more on this later). If you feel you are no longer forgetting this material you can move it into the memorized category. 
</p>    

------------------------------------
    

   

## 2. Main Screen

<p> Some useful nomenclature is the difference between classes and subjects. Classes are the main categories, while Subjects are the categories within the classes. For example if I was taking a class on Neurolgy I may choose to make a class called Neurology. I then would proceed to make subjects on the different lessons taught in that class such as Basal Ganglion, I then would add flashcards to this subject as needed and test on them to learn this subject.    

<ul>
<li><b>Refresh Button</b> - will refresh the subject list screen (useful after adding classes or subjects)</li><p></p>
    
<li><b>Number</b> - will display how many are in the Subject List Screen </li><p></p>
    
<li><b>Event Select</b> -selects between the various displays for the Subject List Screen</li>
        <ul><li><b>Today</b> - Shows the subjects with the lowest guessed forgetting value the amount depends on the Goals value in settings</li>
            <li><b>Learning</b> - Shows the subjects that are currently being learned</li>
            <li><b>Forgetting Test Score</b> - Shows the last test score of the subjects that are being forgotten</li>
            <li><b>Forgetting Estimated Score</b> - Shows the estimated true value of the subject. This value is based on either routine you can change or it is based on Machine learning depending on how much data you collected and if you have run the machine learning program.</li>
            <li><b>Memorized</b> - Shows subjects that are memorized.</li>
            <li><b>Everything</b> - Shows every subject</li>
        </ul><p></p>
<li><b>Subject List Screen</b> - will display the subjects based on the Event select. This is also used to select subjects</li><p></p>

<li><b>Class Select</b> - will filter which classes the subject list screen will display</li><p></p>

<li><b>Take Test Button</b> - starts a test of the subject selected in the Subject list screen</li><p></p>

<li><b>Curve Display</b> - shows either learning curve or forgetting curve depending on the subject</li>
    <ul><li><b>Note: </b> The forgetting curve is always an estimation until a test is taken then the real values are derived from your scores based on the formula $exp^{-t/s}$</li></ul> 
<p></p>

<li><b>Latest Score and days</b> these will tell you the subjects last score and the number of days till you should test on the material. This is calculated based on when it is estimated it will hit your maintenance level which can be changed in settings</li><p></p>

<li><b>s: </b>This is the current s value of the selected subject, this is used in the forgetting equation $exp^{-t/s}$ to determine how quickly the information will be forgotten. This can be changed by clicking on the up or down windows and hitting enter changing when the information is estimated to be forgotten </li><p></p>

<li><b>Add Material</b> This button will open the Add Material window which allows you to add flashcards to the selected subject (which will be discussed later)</li>

<li><b>Learning-Forgetting-Memorized buttons</b> pressing these will move the selected subject into the proper category</li>

<li><b>Edit Material</b> This button will open the Edit Material window which will show the flashcards for the subject and help you edit them (which will be discussed later)


</ul>
        
        
      
         

<p></p>

![image.png](attachment:image.png)


--------------------

## 3. File
<p>The File dropdown menu contains resources to add and remove classes, subjects, and add focuses</p>

![Screen%20Shot%202021-06-05%20at%207.05.40%20PM.png](attachment:Screen%20Shot%202021-06-05%20at%207.05.40%20PM.png)

--------------------------------

<ul><li><b>Add Class</b> - Simply type the name of the class you want to add and you can hit Enter or hit return to create the class</li><p></p>
    
![Screen%20Shot%202021-06-05%20at%207.09.54%20PM.png](attachment:Screen%20Shot%202021-06-05%20at%207.09.54%20PM.png)

--------------

<li><b>Add Subject</b> - Select the class you would like to add the subject to. You can also opt to add a color (refered to as a focus) by checking the box "Automatically place in Focus" and having it set to the desired color. <b>Note: </b> This color will automically be set to the most recent color choice. Type the name of the new subject and click enter or hit return</li><p></p>

![Screen%20Shot%202021-06-05%20at%207.41.29%20PM.png](attachment:Screen%20Shot%202021-06-05%20at%207.41.29%20PM.png)

-----------

<li><b>Add Focus</b> -Allows you to add or remove a color from a subject. Classes are selected on the top left dropdown menu. Subjects are shown in the left screen. Color groups are selected by the dropdown window on the right. Select a subject to add to the current color group and click add or hit enter. To remove a subject from a color group simply select the subject in the color group screen on the right and click remove.</li><p></p> 

![Screen%20Shot%202021-06-06%20at%208.43.50%20AM.png](attachment:Screen%20Shot%202021-06-06%20at%208.43.50%20AM.png)

----------

<li><b>Remove Class and Subject</b> -these follow the same format as the Add Class and Subject windows</li><p></p>
    
----------  
<b>Recent Additions</b>
    
----------
    
<li><b>Rename Class</b> -Select Class and type in the new class name and hit rename</li>
<li><b>Rename Subject</b> -Select Class and Subject and type in the new subject name and hit rename</li>
<li><b>Move Cards</b> -The Class and Subject selectors on the left are the FROM selection and the Class and Subject selectors on the right are the TO selection. Select the cards from the Table you want to move from the table on the left. Once the move cards button is hit these should appear in the table on the left </li><p></p></ul>
        

-------


## 4. Settings
<p>Inside the settings dropdown menu there are two options <b>Change Settings</b> and <b>Defulat time lapse</b> </p> 
<br>
<br>
<center><b>Change Settings</b></center>

------

<p><b>Goals</b> this value is for how many subjects you want to appear in the Today event</p>
<p><b>Tracker</b> this is how many youve tested on today so far. You shouldnt need to change this one manually but option is there in settings </p>

<p><b>Maintenance Level</b> this is the score you would like to maintain your subjects at. Subjects that are below this score will appear red in the Subject List Screen</p>

<p><b>Text Size</b> this will increase the size of the text in the Subject List Screen and testing Window. <b>this requires the application to be restarted to take effect</b></p>


    
<br>
<br>
<center><b>Default time lapse</b></center>

------
<p>This setting is important before you have enough data to run Machine Learning. It will set up your default curve for every subject. Moving the toggles will generate the curves and the days till you reach your maintenance levels. Shown in the image below </p>





![image.png](attachment:image.png)

## 5. Progress

<p>This window will show you the total score and the estimated score of each class based on the subjects of the class</p>

![image.png](attachment:image.png)

## 6. Machine Learning
<p>This section is for running when you have taken enough tests. Everytime you take a test on subjects that are in the forgetting section. Data is collected about the subject. How many tests it took to learn the subject, how many forgetting tests youve taken, how many cards are in the subject, which class it is in, your average score with the std deviation, time to learn, average time to learn and std deviation of that time. and the correct s value that is calculated when you take the test (more on that later)</p><br>

<p>Hit the initiate learning button to run machine learning algorithm. It will show you the testing actual values and the predicted values and the MAE to see how accurate the fitting it. s values for the forgetting curve can range widely and the amount of data collected is minimal compared to all the variables associated with learning. But I have had some great success with this system and MAE as low as 50 I see as a great success but I have also ran it with MAE as high as 400 and still seen relatively closs results. As you collect more data the more accurate this fitting will become. I do forsee updates coming to this section to try to make the algorithm even better </p> 

![Screen%20Shot%202021-07-11%20at%202.09.01%20PM.png](attachment:Screen%20Shot%202021-07-11%20at%202.09.01%20PM.png)

## 7. Data

<p><b>Back Up Data</b> here you can back up your entire data file. simply select where you would like to store the back up (I personally keep mine on my desktop) and it will save all the files needed for this program in a new fodler called BackUp_Day_Month_Year</p><br>

<p><b>Share Data</b> here you can share a subject or class with others. The first data navigation will allow you yo select the subject or file you want to share. The next is where you would like to store that folder (I usually select the desktop </p>

<p><b>Import Data</b> This allows you to import those folders generated from the share data and back up data options. </p>

-------

## 8. Edit Material Screen


<p>This is your one stop shop for editing the cards in your subject. You can also use it to study since every card that youve gotten wrong more times than correct will be highlighted red. Select either the term or definition you would like to change. The text will appear in the window at the bottom which can be adjusted and permently altered by clicking edit. To remove a card simply select the term and hit the remove button. Another neat feature you can double click cells that are .png to see the image</p>


![Screen%20Shot%202021-07-11%20at%202.30.02%20PM.png](attachment:Screen%20Shot%202021-07-11%20at%202.30.02%20PM.png)

--------

## 9. Adding Cards

<p>This is were you can add cards to a subject. Insert the term and definition and click the add material button. Or a shortcut is hitting shift+right key will add the card as is. The boxes with the triple dots allow you to add a special card supported file types are .png, .mov, .mp3, .MP3, .mp4 (.obj and .html will be supported in the future)</p>

![Screen%20Shot%202021-07-11%20at%202.35.31%20PM.png](attachment:Screen%20Shot%202021-07-11%20at%202.35.31%20PM.png)


-----------------

## 10. Taking Tests
<p> The testing window. Hit Start to start the test. When you are finished you can hit this button again (Score) to record the results. Hitting correct or wrong will move the test along (shortcuts are shift+right key for correct and shift+left key for wrong) hitting the definition/term button will flip the card (shortcut is shift+down key). If simply wrong or correct isnt enough you can enter correct number into the lower box on the right and enter wrong number on the lower box on the left then hit correct, this case will be calculated as the percent corect box / wrong box. At the top the card you are on out of the total will be displayed as well </p>



![Screen%20Shot%202021-07-11%20at%202.38.14%20PM.png](attachment:Screen%20Shot%202021-07-11%20at%202.38.14%20PM.png)
