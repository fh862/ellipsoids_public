using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using UnityEngine;
using System;
using UnityEditor;
using UnityEngine.InputSystem;

// Subclass of Expt_4d_color_discrimination.
// This 4D color discrimination experiment reuses many variables and methods from the original 4D color 
// discrimination base class.
// The key differences in behavior are implemented through the subclass’s own private variables and methods.
public class Expt_adaptation : Expt_color_discrimination
{
    public Camera mainCamera;
    /// <overview>
    /// Most of the variables and methods used in this class are inherited from 
    /// the base class `Expt_4d_color_discrimination` via its protected members, 
    /// which allows direct access and reuse in this subclass without redefinition.
    ///
    /// The three private variables defined here — 
    /// (1) gammaCorrection_fileDate_adaptation, 
    /// (2) gammaCorrectionFile_adaptation, and 
    /// (3) durations_adaptation — 
    /// are intentionally separate from the base class’s private counterparts. 
    /// They store experiment-specific values for the thresholding task, which differ 
    /// from the oddity task in gamma correction settings and trial timing configuration.

    // 🔧 Gamma correction: use a more recent calibration measurement
    // The base class defines `gammaCorrection_fileDate` and `gammaCorrectionFile` as private,
    // which means they are not visible or accessible in this subclass.
    // Although we could reuse the same variable names here, we intentionally use distinct names
    // (`gammaCorrection_fileDate_adaptation` and `gammaCorrectionFile_adaptation`) to explicitly
    // differentiate the gamma correction measurements used in the thresholding experiment.
    private string gammaCorrection_fileDate_blob = "02012026"; //"10062025" for the first round of data collection
    private string gammaCorrectionFile_blob;
    private string gammaCorrection_fileDate_background = "02102026"; //"10082025" for the first round of data collection
    private string gammaCorrectionFile_background;
    private bool gammaCorrection_background = true;
    private GammaCorrectionHelper gammaCorrectionHelper_blob;
    private GammaCorrectionHelper gammaCorrectionHelper_background;
    private CameraBackgroundChanger cameraBackgroundChanger;
    private int countDownNumber;
    private bool flagJustHadABreak = false;
    // ⏱️ Trial Timing Durations (in seconds)
    // Unlike the 4D color discrimination experiment, the thresholding experiment does not include
    // correctness-based feedback, as there is no objectively right or wrong answer.
    // Therefore, the "feedback" duration is omitted from this timing configuration.
    private readonly Dictionary<string, float> durations_adaptation = new Dictionary<string, float>
    {
        { "fixation", 0.5f },
        { "blank", 0.2f },
        { "stimulus", 1.5f },
        { "ITI", 1.5f }
    };

    private bool takeScreenshotOfStimuli_adaptation;
    private string subfolderName = "4D_Expt_varyingBackground";
    private Color RGB_cubicRoom;
    private Color RGB_background;
    private Color RGB_cubicRoom_corrected;
    private Color RGB_background_corrected;

    /// <summary>
    /// Unity's Start() method, which runs once when the script is first executed.
    /// This method initializes the experiment by setting up subject information, file paths, 
    /// data storage, gamma correction, and helper classes. It also starts the screen setup process.
    void Start()
    {
        try
        {
            // 1️⃣🌱: Set the random seed for trial generation
            // The seed is used to ensure reproducibility of trials, keeping it consistent with Python scripts.
            // The formula varies based on whether it's a practice session:
            // - If it's a practice session → Seed = subject_id * 1000 + session_today
            // - Otherwise → Seed = subject_id * 100 + session_today
            subject_seed = is_practice ? subject_id * 10000 + session_today + 20 : subject_id * 100 + session_today + 20;

            countDownNumber = is_practice ? 10 : 120;

            takeScreenshotOfStimuli_adaptation = is_practice;

            // 2️⃣📁: Determine file paths for saving experiment data
            // This sets the correct save directory and initialization file path based on subject ID, session number, and whether it's a practice session.
            DetermineFilePaths(subject_id, subject_init, session_today, is_practice, subfolderName);

            // 3️⃣🎨: Load gamma correction data if enabled
            // If gamma correction is enabled, it loads the correction data from the specified file.
            if (gammaCorrection)
            {
                gammaCorrectionFile_blob =
                    "D:/Aguirre-Brainard Lab Dropbox/Fangfang Hong/ELPS_materials/Calibration/" +
                    $"Inverse_gamma_functions_{gammaCorrection_fileDate_blob}/" +
                    $"DELL_{gammaCorrection_fileDate_blob}_texture_right.csv";

                // Check if the gamma correction file exists; if not, throw an error
                if (!File.Exists(gammaCorrectionFile_blob))
                {
                    Debug.LogError($"❌ Error: The file '{gammaCorrectionFile_blob}' does not exist.");
                    return; // Stop execution safely
                }
                gammaCorrectionHelper_blob = new GammaCorrectionHelper(gammaCorrectionFile_blob);
                gammaCorrectionHelper_blob.LoadGammaCorrectionData();
            }

            if (gammaCorrection_background)
            {
                gammaCorrectionFile_background =
                    "D:/Aguirre-Brainard Lab Dropbox/Fangfang Hong/ELPS_materials/Calibration/" +
                    $"Inverse_gamma_functions_{gammaCorrection_fileDate_background}/" +
                    $"DELL_{gammaCorrection_fileDate_background}_background.csv";

                // Check if the gamma correction file exists; if not, throw an error
                if (!File.Exists(gammaCorrectionFile_background))
                {
                    Debug.LogError($"❌ Error: The file '{gammaCorrectionFile_background}' does not exist.");
                    return; // Stop execution safely
                }
                gammaCorrectionHelper_background = new GammaCorrectionHelper(gammaCorrectionFile_background);
                gammaCorrectionHelper_background.LoadGammaCorrectionData();
            }

            // 4️⃣📰: Create a CSV file for recording trial data
            // This ensures that each session has its own log file to store responses and trial information.
            createTrialData_adaptation();

            // 5️⃣🎮: connet to the game pad
            ConnectGamepads();

            // 6️⃣🤝: Initialize helper classes
            // These helper classes assist with color manipulation, file handling, and trial management.
            surfaceColorChanger = new SurfaceColorChanger(gammaCorrectionHelper_blob, gammaCorrection, smoothness); // Handles color adjustments
            cameraBackgroundChanger = new CameraBackgroundChanger(gammaCorrectionHelper_background, gammaCorrection_background);
            fileDataExtractor = new FileDataExtractor(); // Handles reading and writing experiment data
            trialHelper = new TrialHelper(subject_seed); // Generates randomized trial sequences based on subject seed

            // 7️⃣🔁: Start screen setup and experiment initialization
            // This coroutine ensures that the initialization file is correctly set up before starting the experiment.
            StartCoroutine(Check_initializationFile_startSettingup());

            // update the background
            StartCoroutine(Update_Background());
        }
        catch (Exception e)
        {
            Debug.LogError($"Error in Update: {e.Message}");
            UnityEditor.EditorApplication.isPlaying = false; // Stop execution
        }
    }

    /// <summary>
    /// Unity's Update() method, which runs once per frame.
    /// This function continuously checks the last line of an external file to determine the current experiment state.
    /// Based on the detected state, it initiates the appropriate coroutine (trial execution, break handling, 
    /// or experiment termination).

    void Update() // 🔁 Runs every frame
    {
        try
        {
            if (!isExptDone && !isRunningStimulus) // ✅ Only process if no stimulus is running
            {
                // Read the most recent line from the experiment initialization file
                string lastLine = fileDataExtractor.ReadLastLine(initFilePath);
                Debug.Log($"Last read line: {lastLine}"); // ✅ Log the last read line

                // Check if the last line contains specific trigger words
                var (isDone, _) = fileDataExtractor.CheckLastNthCharacters(lastLine, "Done", 0);
                var (isDisplay, _) = fileDataExtractor.CheckLastNthCharacters(lastLine, "Image_Display", 0);
                var (isBreak, _) = fileDataExtractor.CheckLastNthCharacters(lastLine, "Break", 0);
                var (isChangeBackground, _) = fileDataExtractor.CheckLastNthCharacters(lastLine, "Change_Background", 0);

                // Ensure we are not already running a stimulus (prevents duplicate trials)
                if (isDone) // 🔚 If "Done" is detected, terminate the experiment
                {
                    isExptDone = true; // Flag the experiment as completed
                    StartCoroutine(CloseScreen()); // Start the closing procedure
                }
                else if (isDisplay) // 🔛 If "Image_Display" is detected, check for duplicates
                {
                    if (lastLine == lastImageDisplayCommand) // 🚨 Prevent duplicate trials
                    {
                        Debug.Log("⚠ Duplicate 'Image_Display' command detected. Skipping trial.");
                        return; // Exit early to avoid re-running the same trial
                    }

                    // ✅ Store the new Image_Display command
                    lastImageDisplayCommand = lastLine;

                    isRunningStimulus = true; // Prevent overlapping execution
                    StartCoroutine(HandleStimulusDisplay_varyingBackground(lastLine)); // Process and display the stimulus for a new trial
                }
                else if (isBreak) // ⏸ If "Break" is detected, enter the break phase
                {
                    isRunningStimulus = true;
                    StartCoroutine(ResumeExperiment_varyingBackground()); // Wait for the participant to resume the experiment
                    flagJustHadABreak = true;
                }
                else if (isChangeBackground)
                {
                    isRunningStimulus = true;
                    StartCoroutine(Update_Background()); // update the background
                }
                else
                {
                    Debug.Log("No recognized command found. Waiting for next frame...");
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"Error in Update: {e.Message}");
            EditorApplication.isPlaying = false; // Stop execution
        }
    }

    /// <summary>
    /// Handles the sequence of stimulus presentation, post-response feedback, and pre-stimulus setup.
    /// This ensures all necessary steps are completed before the next trial starts.
    /// lastLine: The last recorded line in the initialization file.
    private IEnumerator HandleStimulusDisplay_varyingBackground(string lastLine)
    {
        try
        {
            if (trialCounter == 0 | flagJustHadABreak)
            {
                // Step 0: Run pre-stimulus presentation and WAIT until it finishes
                yield return StartCoroutine(NON_StimulusPresentation_PreStim());
                flagJustHadABreak = false;
            }
            // Step 1: Run stimulus presentation and WAIT until it finishes
            yield return StartCoroutine(StimulusPresentation_varyingBackground(lastLine));

            // Step 2: Run post-response presentation and WAIT until it finishes
            yield return StartCoroutine(NON_StimulusPresentation_postResp());

            // Step 3: Run pre-stimulus presentation and WAIT until it finishes
            yield return StartCoroutine(NON_StimulusPresentation_PreStim());
        }
        finally
        {
            isRunningStimulus = false; // Ensure the flag is always reset
        }
    }

    public IEnumerator ResumeExperiment_varyingBackground()
    {
        displayText.text = "Break time";
        // Wait until either the resume key or gamepad buttonSouth is pressed
        yield return new WaitUntil(() =>
        {
            if (Input.GetKeyDown(ResumeKey)) return true;
            if (gamepad != null && gamepad.buttonSouth.wasPressedThisFrame) return true;
            return false;
        });

        // Write 'resume' in the txt file so that the Python side knows
        fileDataExtractor.WriteMessageToFile(initFilePath, "Resume");

        displayText.text = "Resume";
        yield return new WaitForSeconds(1.0f);

        isRunningStimulus = false;
    }

    private IEnumerator Update_Background()
    {
        displayText.text = "Press ↓ to start";
        yield return new WaitForSeconds(.5f);

        //change the background
        string command_changebg = "Change_Background";
        // This is a while loop. The default max wait time is 300 s to avoid indefinite loop
        yield return StartCoroutine(fileDataExtractor.VerifyFileExistsAndCheckNthLastCharacters(initFilePath, command_changebg, 0));
        string lastLine_changebg = fileDataExtractor.ReadLastLine(initFilePath);

        // 📄 Extract trial information and reference/comparison stimulus colors from the file
        var (BackgroundColorString_new, CubicRoomColorString_new) = fileDataExtractor.ExtractBackgroundCubicRoom(lastLine_changebg);

        // 🎨 Extract RGB values for the reference and comparison stimuli
        var (bg_R_new, bg_G_new, bg_B_new) = fileDataExtractor.ExtractRGBValues(BackgroundColorString_new);
        var (cr_R_new, cr_G_new, cr_B_new) = fileDataExtractor.ExtractRGBValues(CubicRoomColorString_new);

        // ✅ Store raw parsed colors
        RGB_cubicRoom = new Color(cr_R_new, cr_G_new, cr_B_new);
        RGB_background = new Color(bg_R_new, bg_G_new, bg_B_new);

        // Set room color
        surfaceColorChanger.ChangeSurfaceTextureColor(RendererTopObj_room, cr_R_new, cr_G_new, cr_B_new);
        surfaceColorChanger.ChangeSurfaceTextureColor(RendererLeftObj_room, cr_R_new, cr_G_new, cr_B_new);
        RGB_cubicRoom_corrected = surfaceColorChanger.ChangeSurfaceTextureColor(RendererRightObj_room, cr_R_new, cr_G_new, cr_B_new);

        // Camera's background color
        RGB_background_corrected = cameraBackgroundChanger.ChangeCameraBackground(mainCamera, bg_R_new, bg_G_new, bg_B_new);

        // Wait until either the resume key or gamepad buttonSouth is pressed
        yield return new WaitUntil(() =>
        {
            if (Input.GetKeyDown(ResumeKey)) return true;
            if (gamepad != null && gamepad.buttonSouth.wasPressedThisFrame) return true;
            return false;
        });

        // Countdown from N to 1
        for (int i = countDownNumber; i > 0; i--)
        {
            displayText.text = i.ToString();
            yield return new WaitForSeconds(1.0f);
        }

        // Write a message to the txt file to indicate the completion of the setup
        fileDataExtractor.WriteMessageToFile(initFilePath, "Change_Background_Confirmed");
        Debug.Log("Background changes completed.");

        isRunningStimulus = false;
    }

    /// <summary>
    /// Handles the presentation of visual stimuli and records the participant's response.
    /// last_line: The last recorded trial data from the experiment file.
    private IEnumerator StimulusPresentation_varyingBackground(string last_line)
    {
        // Compute actual wait time after '+' disappears by taking the delay of reading files into account
        yield return new WaitForEndOfFrame(); // Ensure frame update happens
        if (timeStamp_preStim_BlankScreen > 0)
        {
            actualWait_preStim = Time.time - timeStamp_preStim_BlankScreen;
        }
        else
        {
            actualWait_preStim = 0.0f;
        }

        // Determine the odd object's location using the trial sequence generator
        int OddLocation = trialHelper.RefCompRandomSequence();

        // 📄 Extract trial information and reference/comparison stimulus colors from the file
        var (TrialInfo, RefColorString, CompColorString) = fileDataExtractor.ExtractTrialRefComp(last_line);

        // 🎨 Extract RGB values for the reference and comparison stimuli
        var (ref_R, ref_G, ref_B) = fileDataExtractor.ExtractRGBValues(RefColorString);
        var (comp_R, comp_G, comp_B) = fileDataExtractor.ExtractRGBValues(CompColorString);

        // 👀 Make stimuli visible
        RendererTopObj.enabled = true;
        RendererLeftObj.enabled = true;
        RendererRightObj.enabled = true;

        // Initialize variables for gamma-corrected RGB values
        Color RGB_ref_corrected = new Color(0f, 0f, 0f);
        Color RGB_comp_corrected = new Color(0f, 0f, 0f);

        // 🔄 Display stimuli with gamma correction applied (based on OddLocation)
        if (OddLocation == 1)
        {
            RGB_comp_corrected = surfaceColorChanger.ChangeSurfaceTextureColor(RendererTopObj, comp_R, comp_G, comp_B);
            RGB_ref_corrected = surfaceColorChanger.ChangeSurfaceTextureColor(RendererLeftObj, ref_R, ref_G, ref_B);
            surfaceColorChanger.ChangeSurfaceTextureColor(RendererRightObj, ref_R, ref_G, ref_B);
        }
        else if (OddLocation == 2)
        {
            RGB_comp_corrected = surfaceColorChanger.ChangeSurfaceTextureColor(RendererLeftObj, comp_R, comp_G, comp_B);
            RGB_ref_corrected = surfaceColorChanger.ChangeSurfaceTextureColor(RendererTopObj, ref_R, ref_G, ref_B);
            surfaceColorChanger.ChangeSurfaceTextureColor(RendererRightObj, ref_R, ref_G, ref_B);
        }
        else if (OddLocation == 3)
        {
            RGB_comp_corrected = surfaceColorChanger.ChangeSurfaceTextureColor(RendererRightObj, comp_R, comp_G, comp_B);
            RGB_ref_corrected = surfaceColorChanger.ChangeSurfaceTextureColor(RendererTopObj, ref_R, ref_G, ref_B);
            surfaceColorChanger.ChangeSurfaceTextureColor(RendererLeftObj, ref_R, ref_G, ref_B);
        }

        // 📷 Take a screenshot after the first frame update (only once per iteration)
        if (takeScreenshotOfStimuli_adaptation && !screenshotTaken)
        {
            yield return new WaitForEndOfFrame(); // Let one full frame pass with the stimuli visible
            string filename = Path.Combine(
                saveDirectory,
                $"Session{session_today}_" +
                $"Ref_R{ref_R}_G{ref_G}_B{ref_B}_" +
                $"Comp_R{comp_R}_G{comp_G}_B{comp_B}.png"
            );
            ScreenCapture.CaptureScreenshot(filename);
            screenshotTaken = true;  // Prevent multiple screenshots in the same iteration
            Debug.Log($"📷 Screenshot saved: {filename}");
            yield return new WaitForEndOfFrame();  // Ensure screenshot is fully processed
        }
        screenshotTaken = false;  // Reset flag for next iteration

        // 🍡 Keep the stimuli on screen for the duration of the stimulus presentation
        yield return new WaitForSeconds(durations_adaptation["stimulus"]);

        // 🙈 Hide stimuli after presentation ends
        RendererTopObj.enabled = false;
        RendererLeftObj.enabled = false;
        RendererRightObj.enabled = false;

        // Convert RGB values to string format for logging and storage
        string RGB_ref_string = fileDataExtractor.ConvertRGBToString(ref_R, ref_G, ref_B);
        string RGB_comp_string = fileDataExtractor.ConvertRGBToString(comp_R, comp_G, comp_B);
        string RGB_background_string = fileDataExtractor.ConvertColorToString(RGB_background);
        string RGB_cubicRoom_string = fileDataExtractor.ConvertColorToString(RGB_cubicRoom);
        string RGB_ref_corrected_string = fileDataExtractor.ConvertColorToString(RGB_ref_corrected);
        string RGB_comp_corrected_string = fileDataExtractor.ConvertColorToString(RGB_comp_corrected);
        string RGB_background_corrected_string = fileDataExtractor.ConvertColorToString(RGB_background_corrected);
        string RGB_cubicRoom_corrected_string = fileDataExtractor.ConvertColorToString(RGB_cubicRoom_corrected);

        // 🎮 Capture the participant's response
        yield return StartCoroutine(GetResponse(OddLocation));

        // 💾 Log trial data, including stimulus parameters and participant response
        RegisterResponse_adaptation(OddLocation, TrialInfo, RGB_ref_string, RGB_comp_string,
            RGB_background_string, RGB_cubicRoom_string, RGB_ref_corrected_string, RGB_comp_corrected_string,
            RGB_background_corrected_string, RGB_cubicRoom_corrected_string, codedAnswer, pressedKeysString,
            actualWait_preStim, responseTime);

        // ➕ Move to the next trial
        trialCounter++;
    }


    /// <summary>
    /// Logs the participant's response and stores trial-related data.
    /// This function ensures that AEPsych receives the response immediately 
    /// to compute the next trial efficiently while also recording the data for analysis.
    /// trialCounter        : Current trial number (starting from 0)
    /// trialInfo           : Information about the trial (e.g., Trial_10_Cond_1)
    /// Comp1Location       : Position of the fixed comparison stimulus (1 = bottom left, 2 = bottom right)
    /// pressedKeys         : String representation of the response keys pressed (e.g., "1_0", "0_1")
    /// RGB_ref             : RGB values of the reference stimulus before correction
    /// RGB_comp1           : RGB values of the fixed comparison stimulus before correction
    /// RGB_background      : RGB values of the comparison stimulus that changes 
    /// RGB_ref_corrected   : Corrected RGB values of the reference stimulus
    /// RGB_comp1_corrected : Corrected RGB values of the fixed comparison stimulus
    /// RGB_background_corrected : Corrected RGB values of the varying comparison stimulus
    /// codedResp           : Binary-coded response (1 = Correct, 0 = Incorrect)
    /// RT                  : Reaction time (in seconds)
    /// actual_wait_preStim : The time between the disappearance of '+' and stimulus presentation
    /// LUT                 : look-up table for gamma correction
    private void RegisterResponse_adaptation(int CompLocation, string TrialInfo, string ref_RGB,
        string comp_RGB, string background_RGB, string cubicRoom_RGB, string ref_RGB_corrected,
        string comp_RGB_corrected, string background_RGB_corrected, string cubicRoom_RGB_corrected,
        float binaryResp, string pressedKeys, float actual_wait_preStim, float respTime)
    {
        // 💾 Log response immediately for real-time trial adaptation
        // This allows AEPsych to compute the next trial placement as soon as possible.
        fileDataExtractor.WriteResponseToFile(initFilePath, TrialInfo, ref_RGB, comp_RGB, codedAnswer);

        // 💾 Append trial data for long-term storage & analysis
        // This ensures the collected responses are saved in a structured CSV file.
        AppendTrialData_adaptation(
            trialCounter.ToString(),
            TrialInfo,
            CompLocation.ToString(),
            pressedKeys,
            ref_RGB,
            comp_RGB,
            background_RGB,
            cubicRoom_RGB,
            ref_RGB_corrected,
            comp_RGB_corrected,
            background_RGB_corrected,
            cubicRoom_RGB_corrected,
            binaryResp.ToString(),
            actual_wait_preStim.ToString(),
            respTime.ToString()
        );
    }

    private void createTrialData_adaptation()
    {
        // Generate timestamp in the format YYYY-MM-DD_HH-MM-SS
        string timeStamp = DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss");

        // Construct the file path with timestamp
        csvFilePath = Path.Combine(saveDirectory, $"Unity_trial_data_{filePrefix}_{timeStamp}.csv");

        // If the file does not exist, write the header
        if (!File.Exists(csvFilePath))
        {
            using (StreamWriter sw = new StreamWriter(csvFilePath, true))
            {
                sw.WriteLine("TrialCounter,TrialInfo, CompLocation, PressedKey, Ref, Comp, "+
                    "Background, CubicRoom, Ref_Corrected, Comp_Corrected, "+
                    "Background_Corrected, CubicRoom_Corrected, binaryResp, actual_wait_preStim, RT, " +
                    $"subID_{subject_id.ToString()}, subInit_{subject_init}, sessNum_{session_today.ToString()}, " +
                    $"seed_{subject_seed.ToString()}, LUT_{gammaCorrectionFile_blob}, "+
                    $"LUT_{gammaCorrectionFile_background}, countdown_{countDownNumber}_seconds");
            }
        }
    }

    private void AppendTrialData_adaptation(
        string trialCounter, string trialInfo, string OddLocation, string pressedKeys,
        string RGB_ref, string RGB_comp, string RGB_background, string RGB_cubicRoom,
        string RGB_ref_corrected, string RGB_comp_corrected, string RGB_background_corrected,
        string RGB_cubicRoom_corrected, string codedResp, string actual_wait_preStim, string RT)
    {
        using (StreamWriter sw = new StreamWriter(csvFilePath, true))
        {
            // Write to CSV file with better formatting (split into two lines for readability)
            sw.WriteLine(
                $"{trialCounter}, {trialInfo}, {OddLocation}, {pressedKeys}, {RGB_ref}," +
                $"{RGB_comp}, {RGB_background}, {RGB_cubicRoom}, {RGB_ref_corrected}, "+
                $"{RGB_comp_corrected}, {RGB_background_corrected}, {RGB_cubicRoom_corrected}, "+
                $"{codedResp}, {actual_wait_preStim}, {RT}"
            );
        }
    }

}
