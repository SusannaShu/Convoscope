import time
import traceback
import asyncio

# custom
from DatabaseHandler import DatabaseHandler
from agents.language_learning_agent import run_language_learning_agent
from agents.helpers.get_dictionary_rank import get_dictionary_rank

run_period = 3

def language_learning_agents_processing_loop():
    print("START LANGUAGE LEARNING PROCESSING LOOP")
    dbHandler = DatabaseHandler(parent_handler=False)
    loop = asyncio.get_event_loop()

    while True:
        if not dbHandler.ready:
            print("dbHandler not ready")
            time.sleep(0.1)
            continue

        # wait for some transcripts to load in
        time.sleep(1)

        try:
            pLoopStartTime = time.time()
            # Check for new transcripts
            print("RUNNING LANGUAGE LEARNING LOOP")
            newTranscripts = dbHandler.get_recent_transcripts_from_last_nseconds_for_all_users(
                n=run_period*2)

            words_to_show = None
            for transcript in newTranscripts:
                print(transcript)
                ctime = time.time()
                dictionary_rank = get_dictionary_rank(transcript['text'])
                #get users target language
                target_language = dbHandler.get_user_option_value(transcript['user_id'], "target_language")
                print("GOT LANGUAGE: " + target_language)

                #run the language learning agent
                words_to_show = run_language_learning_agent(transcript['text'], dictionary_rank, target_language)
                loop_time = time.time() - ctime
                print(f"RAN LL IN : {loop_time}")
                print(words_to_show)

                if words_to_show:
                    final_words_to_show = list(filter(None, words_to_show))
                    print("WORDS TO SHOW")
                    print(final_words_to_show)
                    dbHandler.add_language_learning_words_to_show_for_user(transcript['user_id'], final_words_to_show)

        except Exception as e:
            print("Exception in Language learning loop...:")
            print(e)
            traceback.print_exc()

        finally:
            # lock.release()
            pLoopEndTime = time.time()
            print("=== language leanring loop completed in {} seconds overall ===".format(
                round(pLoopEndTime - pLoopStartTime, 2)))

        time.sleep(run_period)
