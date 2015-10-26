Notes about coding style
-----------------------

As noted by one of your colleagues after the tutorial, good coding style is
fundamental. It is specially important when someone has to mark your code. To
help yourself write pretty Python code, your anonymous style-hero classmate and
I recommend the package `flake8`, that you can easily pip inside your virtual
environment:

```
pip install flake8
```

Now you can run the command `flake8` on any Python script and it will tell you
where and why it looks ugly, according the the world-famous **PEP8 Style
Guide**. PEP8 is a standard of good coding practice and should be respected,
particularly in collaborative projects.

That said, I politely disagree with some of the conventions in PEP8, and I have
deliberately chosen to not follow them in the exercise scripts. These are the
specific PEP8 sections I respectfully ignore:

*Disclaimer*: all the lines below start with "I" because it's all my personal
preference.  But those are like opinions --- everybody has one. You don't have
to stare at mine for any longer than you deem appropriate.

1. I prefer indentations with 2 spaces rather than 4.

2. I think it's a good idea to leave an empty line at the end of a script.

3. I'm ok with leaving extra spaces at the sides of operators to align several
statements.

4. I tend to begin code "sections" with double-hashed comments, i.e. coments
starting with '## '. This disagrees with the PEP8 mantra that forces all
comments to start with '# '.

To stop flake8 from complaining all the time about these, I style-check my
python scripts with the following command:

```
flake8 myFancyScript.py | grep -Ev 'E221|E111|E265|W391'
```

**Some marks of the assessed coursework are reserved for style and clarity**.
We won't strictly enforce all PEP8 guidelines, but it's probably a good idea
to try to follow it.


