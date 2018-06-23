
## Description
  - Each top-level directory (*e.g.* `01-resnets`) looks like the following:
    - `papers`: Folders for each paper.
      - `<arxiv_id>`: Folder for a paper.
        - `<arxiv_id>.pdf`: The paper PDF.
        - `<arxiv_id>.md`: Markdown description of paper with summary notes.
    - `<top-level>.apkg`: Anki flashcards for the papers exported to .apgk
    - `anki-collab-<top-level>`: alternative to .apgk that let's you contribute to the Anki deck (see Anki Guide)
    - `code`: Holds code for re-implementing the paper


## Anki Guide

### Motivation to use Anki
Anki utilizes the testing effect and the spacing effect, which arguably are the two most important (and robust) findings in the science of learning.

Michael Nielsen: https://twitter.com/michael_nielsen/status/957763229454774272 

Andrej Karpathy: https://twitter.com/karpathy/status/960556555526524928 

### Getting started

1. Download Anki to your computer  http://ankisrs.net. 
2. Open Anki and create a deck
3. Register at www.ankiweb.net 
4. In Anki on your computer, click on the circle up to the right. Login with your ankiweb credentials. This lets you synch your decks with all your phone and ipad. 
5. Download _______ as an example of a deck.
6. (Optional) Download AnkiDroid (free) to Android or Anki to iPhone (expensive). Synch your phone app to Ankiweb.


### Writing good Anki questions
First, read http://www.supermemo.com/articles/20rules.htm 
Second, go through resnet-01 deck to see example questions.

In general, we aim for 5-12 questions per paper:

- 1 question: who is lead author, what year and what institution?
- 2-4 questions on the problem the paper is adressing
- 2-4 questions on the method
- 1-2 questions on results & findings
- 1-2 questions on limitations
- 1-2 questions on community reaction and impact on the field


#### 

### Collaborating on Anki (advanced)
The .apgk is the standard file format for Anki decks. It is not ideal for collaborating on Anki decks however. 

#### Step 1. Download the CrowdAnki plugin 

In Anki on desktop, click on Tools  --> Add-ons  --> Browse and Install --> "1788670778"

#### Step 2. Read about plugin

CrowdAnki on Ankiweb: https://ankiweb.net/shared/info/1788670778
CrowdAnki on Github: https://github.com/Stvad/CrowdAnki

