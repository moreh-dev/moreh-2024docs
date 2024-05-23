---
icon: terminal
tags: [tutorial, llama2]
order: 40
---

# 4. Checking Training Results

When you execute the **`train_llama3.py`** script as in the previous section, the resulting model will be saved in the **`llama3_summarization`** directory. This model is compatible with any GPU server, not just MoAI Platform, as it is stored as a pure PyTorch model parameter file.

You can test the trained model using the **`inference_llama3.py`** script located under the **`tutorial`** directory in the GitHub repository you downloaded earlier.

For testing, articles related to soldiers deployed in Iraq were used.

```python
# tutorial/inference_llama3.py
...
input_text = """[SUMMARIZE] (CNN) -- A Marine convicted for his role in the death of an Iraqi civilian was sentenced Friday to a reduction in rank and will be discharged. Cpl. Trent D. Thomas was found guilty Wednesday of kidnapping and conspiracy to commit several offenses -- including murder, larceny, housebreaking, kidnapping, and making false official statements -- for his involvement in the April 2006 death in Hamdaniya, Iraq. Thomas will be demoted to the rank of entry-level private and will receive a bad-conduct discharge. The 25-year-old was among seven Marines and a Navy medic who were charged in connection with the death of Hashim Ibrahim Awad, 52. The Marines accused in the case were members of Kilo Company, 3rd Battalion, 5th Marine Regiment. They reported at the time that Awad planned to detonate a roadside bomb targeting their patrol. But several residents of Hamdaniya, including relatives of the victim, gave a different account, prompting a criminal investigation. Prosecutors accuse the group's squad leader, Sgt. Lawrence G. Hutchins III, of dragging Awad from his home, shooting him in the street and then making it look like he had planned to ambush American troops. Hutchins has pleaded not guilty to murder, conspiracy and other charges in the case. He faces a sentence of life in prison if convicted. Thomas changed his plea from guilty to not guilty in February, arguing that he had merely followed orders. He told his attorneys that after reviewing the evidence against him, he realized "that what happened overseas happened as a result of obedience to orders, and he hasn't done anything wrong," defense attorney Victor Kelley said. Thomas said in January, shortly after entering his guilty plea, that he was "truly sorry" for his role in the killing. He could have been sentenced to life in prison under his original plea. E-mail to a friend . [/SUMMAIRZE]"""
```

Run the code.

```bash
~/quickstart$ python tutorial/inference_llama3.py
```

Upon examining the output, you will see that the model appropriately summarizes the contents of the input prompt.

```
Llama3: [SUMMARIZE] (CNN) -- A Marine convicted for his role in the death of an Iraqi civilian was sentenced Friday to a reduction in rank and will be discharged. Cpl. Trent D. Thomas was found guilty Wednesday of kidnapping and conspiracy to commit several offenses -- including murder, larceny, housebreaking, kidnapping, and making false official statements -- for his involvement in the April 2006 death in Hamdaniya, Iraq. Thomas will be demoted to the rank of entry-level private and will receive a bad-conduct discharge. The 25-year-old was among seven Marines and a Navy medic who were charged in connection with the death of Hashim Ibrahim Awad, 52. The Marines accused in the case were members of Kilo Company, 3rd Battalion, 5th Marine Regiment. They reported at the time that Awad planned to detonate a roadside bomb targeting their patrol. But several residents of Hamdaniya, including relatives of the victim, gave a different account, prompting a criminal investigation. Prosecutors accuse the group's squad leader, Sgt. Lawrence G. Hutchins III, of dragging Awad from his home, shooting him in the street and then making it look like he had planned to ambush American troops. Hutchins has pleaded not guilty to murder, conspiracy and other charges in the case. He faces a sentence of life in prison if convicted. Thomas changed his plea from guilty to not guilty in February, arguing that he had merely followed orders. He told his attorneys that after reviewing the evidence against him, he realized "that what happened overseas happened as a result of obedience to orders, and he hasn't done anything wrong," defense attorney Victor Kelley said. Thomas said in January, shortly after entering his guilty plea, that he was "truly sorry" for his role in the killing. He could have been sentenced to life in prison under his original plea. E-mail to a friend . [/SUMMAIRZE]
Cpl. Trent D. Thomas was found guilty of kidnapping and conspiracy .
He will be demoted to the rank of entry-level private and will receive a bad-conduct discharge .
Thomas was among seven Marines and a Navy medic charged in the death .
```