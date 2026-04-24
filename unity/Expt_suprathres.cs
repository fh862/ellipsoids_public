using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using UnityEngine;
using System;
using UnityEditor;
using UnityEngine.InputSystem;

// Subclass of Expt_4d_color_discrimination.
// This suprathreshold experiment reuses many variables and methods from the 4D color discrimination base class.
// The key differences in behavior are implemented through the subclass’s own private variables and methods.
public class Expt_suprathres : Expt_color_discrimination
{
    /// <overview>
    /// Most of the variables and methods used in this class are inherited from 
    /// the base class `Expt_4d_color_discrimination` via its protected members, 
    /// which allows direct access and reuse in this subclass without redefinition.
    ///
    /// The three private variables defined here — 
    /// (1) gammaCorrection_fileDate_suprathres, 
    /// (2) gammaCorrectionFile_suprathres, and 
    /// (3) durations_suprathres — 
    /// are intentionally separate from the base class’s private counterparts. 
    /// They store experiment-specific values for the suprathreshold task, which differ 
    /// from the oddity task in gamma correction settings and trial timing configuration.

    // Not serialized, not shown in Inspector, accessible in this class & subclasses
    protected int RefLocation  { get; private set; }
    protected int Comp1Location { get; private set; }
    protected int Comp2Location { get; private set; }
    private string validKeysString;
    private bool flagJustHadABreak = false;
    // 🔧 Gamma correction: use a more recent calibration measurement
    // The base class defines `gammaCorrection_fileDate` and `gammaCorrectionFile` as private,
    // which means they are not visible or accessible in this subclass.
    // Although we could reuse the same variable names here, we intentionally use distinct names
    // (`gammaCorrection_fileDate_suprathres` and `gammaCorrectionFile_suprathres`) to explicitly
    // differentiate the gamma correction measurements used in the suprathreshold experiment.
    private string gammaCorrection_fileDate_suprathres = "10062025";
    private string gammaCorrectionFile_suprathres;
    // ⏱️ Trial Timing Durations (in seconds)
    // Unlike the 4D color discrimination experiment, the suprathreshold experiment does not include
    // correctness-based feedback, as there is no objectively right or wrong answer.
    // Therefore, the "feedback" duration is omitted from this timing configuration.
    private readonly Dictionary<string, float> durations_suprathres = new Dictionary<string, float>
    {
        { "precue", 0.5f },
        { "blank", 0.2f },
        { "stimulus", 1.5f },
        { "ITI", 1.5f }
    };

    private bool takeScreenshotOfStimuli_suprathres = false;
    private string subfolderName = "supra_thres_pilot1";

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
            subject_seed = is_practice ? subject_id * 10000 + session_today : subject_id * 100 + session_today;

            // 2️⃣📁: Determine file paths for saving experiment data
            // This sets the correct save directory and initialization file path based on subject ID, session number, and whether it's a practice session.
            DetermineFilePaths(subject_id, subject_init, session_today, is_practice, subfolderName);

            // 3️⃣🎨: Load gamma correction data if enabled
            // If gamma correction is enabled, it loads the correction data from the specified file.
            if (gammaCorrection)
            {
                gammaCorrectionFile_suprathres =
                    "D:/Aguirre-Brainard Lab Dropbox/Fangfang Hong/ELPS_materials/Calibration/" +
                    $"Inverse_gamma_functions_{gammaCorrection_fileDate_suprathres}/" +
                    $"DELL_{gammaCorrection_fileDate_suprathres}_texture_right.csv";

                // Check if the gamma correction file exists; if not, throw an error
                if (!File.Exists(gammaCorrectionFile_suprathres))
                {
                    Debug.LogError($"❌ Error: The file '{gammaCorrectionFile_suprathres}' does not exist.");
                    return; // Stop execution safely
                }
                gammaCorrectionHelper = new GammaCorrectionHelper(gammaCorrectionFile_suprathres);
                gammaCorrectionHelper.LoadGammaCorrectionData();
            }

            // 4️⃣📰: Create a CSV file for recording trial data
            // This ensures that each session has its own log file to store responses and trial information.
            createTrialData_suprathres();

            // 5️⃣🎮: connet to the game pad
            ConnectGamepads();

            // 6️⃣🤝: Initialize helper classes
            // These helper classes assist with color manipulation, file handling, and trial management.
            surfaceColorChanger = new SurfaceColorChanger(gammaCorrectionHelper, gammaCorrection, smoothness); // Handles color adjustments
            fileDataExtractor = new FileDataExtractor(); // Handles reading and writing experiment data
            trialHelper = new TrialHelper(subject_seed); // Generates randomized trial sequences based on subject seed

            // 7️⃣🔁: Start screen setup and experiment initialization
            // This coroutine ensures that the initialization file is correctly set up before starting the experiment.
            StartCoroutine(Check_initializationFile_startSettingup());
        }
        catch (Exception e)
        {
            Debug.LogError($"Error in Update: {e.Message}");
            EditorApplication.isPlaying = false; // Stop execution
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
                    StartCoroutine(HandleStimulusDisplay_fixedRef(lastLine)); // Process and display the stimulus for a new trial
                }
                else if (isBreak) // ⏸ If "Break" is detected, enter the break phase
                {
                    isRunningStimulus = true;
                    StartCoroutine(ResumeExperiment_suprathres()); // Wait for the participant to resume the experiment
                    flagJustHadABreak = true;
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
    private IEnumerator HandleStimulusDisplay_fixedRef(string lastLine)
    {
        try
        {
            if (trialCounter == 0 | flagJustHadABreak)
            {
                // Countdown from N to 1
                for (int i = 5; i > 0; i--)
                {
                    displayText.text = i.ToString();
                    yield return new WaitForSeconds(1.0f);
                }
        
                // Step 0: Run pre-stimulus presentation and WAIT until it finishes
                yield return StartCoroutine(NON_StimulusPresentation_PreStim_wPreCue());
                flagJustHadABreak = false;
            }

            // Step 1: Run stimulus presentation and WAIT until it finishes
            yield return StartCoroutine(StimulusPresentation_RefComp1Comp2(lastLine));

            // Step 2: Run post-response presentation and WAIT until it finishes
            yield return StartCoroutine(NON_StimulusPresentation_postResp_noFeedback());

            // Step 3: Run pre-stimulus presentation and WAIT until it finishes
            yield return StartCoroutine(NON_StimulusPresentation_PreStim_wPreCue());
        }
        finally
        {
            isRunningStimulus = false; // Ensure the flag is always reset
        }
    }

    /// <summary>
    /// Handles **pre-stimulus events**, ensuring a controlled viewing sequence before stimulus presentation.
    /// - Displays a fixation cross ("+") for 0.5s.
    /// - Shows a brief blank screen (0.2s) before the stimulus appears.
    public IEnumerator NON_StimulusPresentation_PreStim_wPreCue()
    {
        // each time you need a new order:
        List<int> RefComp1Comp2_location = trialHelper.ShuffleRefComp1Comp2();  // e.g., [1,0,2]
        var (r, c1, c2) = trialHelper.DecodeLocations(RefComp1Comp2_location);

        // ✅ assign to fields (no shadowing)
        RefLocation = r + 1;     // convert to 1-based if DecodeLocations returns 0-based
        Comp1Location = c1 + 1;
        Comp2Location = c2 + 1;

        // Generate a precue to indicate the location of the reference
        displayText.text = RefLocation switch
        {
            1 => "↑", // top
            2 => "←", // left
            3 => "→", // right
            _ => ""
        };

        // 📷 Take a screenshot after the first frame update (only once per iteration)
        if (takeScreenshotOfStimuli && trialCounter == 1)
        {
            string filename = Path.Combine(saveDirectory, $"precue.png");
            ScreenCapture.CaptureScreenshot(filename);
            yield return new WaitForEndOfFrame();  // Ensure screenshot is fully processed
        }

        yield return new WaitForSeconds(durations_suprathres["precue"]);

        // Clear text for brief blank screen
        displayText.text = "";

        // 🔹 🕒 **NOW** Record the time between the '+' and stimulus presentation
        // because there might be a slight delay caused by reading files
        yield return new WaitForEndOfFrame();
        timeStamp_preStim_BlankScreen = Time.time;  

        // 📷 Take a screenshot after the first frame update (only once per iteration)
        if (takeScreenshotOfStimuli && trialCounter == 1)
        {
            string filename = Path.Combine(saveDirectory, $"blankscreen.png");
            ScreenCapture.CaptureScreenshot(filename);
            yield return new WaitForEndOfFrame();  // Ensure screenshot is fully processed
        }
        yield return new WaitForSeconds(durations_suprathres["blank"]);
    }

    /// <summary>
    /// Handles **post-response events**, including just the ITI (Inter-Trial Interval).
    private IEnumerator NON_StimulusPresentation_postResp_noFeedback()
    {
        // Clear feedback text
        displayText.text = "";

        // Wait for ITI
        yield return new WaitForSeconds(durations_suprathres["ITI"]);
    }

    public IEnumerator ResumeExperiment_suprathres()
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

    /// <summary>
    /// Handles the presentation of visual stimuli and records the participant's response.
    /// last_line: The last recorded trial data from the experiment file.
    private IEnumerator StimulusPresentation_RefComp1Comp2(string last_line)
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

        // 📄 Extract trial information and reference/comparison stimulus colors from the file
        var (TrialInfo, RefColorString, CompColorString, Comp2ColorString) = fileDataExtractor.ExtractTrialRefComp2(last_line);

        // 🎨 Extract RGB values for the reference and comparison stimuli
        var (ref_R, ref_G, ref_B) = fileDataExtractor.ExtractRGBValues(RefColorString);
        var (comp_R, comp_G, comp_B) = fileDataExtractor.ExtractRGBValues(CompColorString);
        var (comp2_R, comp2_G, comp2_B) = fileDataExtractor.ExtractRGBValues(Comp2ColorString);

        // 👀 Make stimuli visible
        RendererTopObj.enabled = true;
        RendererLeftObj.enabled = true;
        RendererRightObj.enabled = true;

        // 🔄 Display stimuli with gamma correction applied (based on OddLocation)
        var pos = new Dictionary<int, Renderer> {
            {1, RendererTopObj}, {2, RendererLeftObj}, {3, RendererRightObj}
        };
        Color RGB_ref_corrected   = surfaceColorChanger.ChangeSurfaceTextureColor(pos[RefLocation],   ref_R,  ref_G,  ref_B);
        Color RGB_comp_corrected  = surfaceColorChanger.ChangeSurfaceTextureColor(pos[Comp1Location], comp_R, comp_G, comp_B);
        Color RGB_comp2_corrected = surfaceColorChanger.ChangeSurfaceTextureColor(pos[Comp2Location], comp2_R, comp2_G, comp2_B);

        // 📷 Take a screenshot after the first frame update (only once per iteration)
        if (takeScreenshotOfStimuli_suprathres && !screenshotTaken)
        {
            yield return new WaitForEndOfFrame(); // Let one full frame pass with the stimuli visible
            string filename = Path.Combine(
                saveDirectory,
                $"Ref_R{ref_R}_G{ref_G}_B{ref_B}_" +
                $"Comp_R{comp_R}_G{comp_G}_B{comp_B}_" +
                $"Comp2_R{comp2_R}_G{comp2_G}_B{comp2_B}.png"
            );
            ScreenCapture.CaptureScreenshot(filename);
            screenshotTaken = true;  // Prevent multiple screenshots in the same iteration
            Debug.Log($"📷 Screenshot saved: {filename}");
            yield return new WaitForEndOfFrame();  // Ensure screenshot is fully processed
        }
        screenshotTaken = false;  // Reset flag for next iteration

        // 🍡 Keep the stimuli on screen for the duration of the stimulus presentation
        yield return new WaitForSeconds(durations_suprathres["stimulus"]);

        // 🙈 Hide stimuli after presentation ends
        RendererTopObj.enabled = false;
        RendererLeftObj.enabled = false;
        RendererRightObj.enabled = false;

        // Convert RGB values to string format for logging and storage
        string RGB_ref_string = fileDataExtractor.ConvertRGBToString(ref_R, ref_G, ref_B);
        string RGB_comp_string = fileDataExtractor.ConvertRGBToString(comp_R, comp_G, comp_B);
        string RGB_comp2_string = fileDataExtractor.ConvertRGBToString(comp2_R, comp2_G, comp2_B);
        string RGB_ref_corrected_string = fileDataExtractor.ConvertRGBToString(RGB_ref_corrected.r, RGB_ref_corrected.g, RGB_ref_corrected.b);
        string RGB_comp_corrected_string = fileDataExtractor.ConvertRGBToString(RGB_comp_corrected.r, RGB_comp_corrected.g, RGB_comp_corrected.b);
        string RGB_comp2_corrected_string = fileDataExtractor.ConvertRGBToString(RGB_comp2_corrected.r, RGB_comp2_corrected.g, RGB_comp2_corrected.b);

        // 🎮 Capture the participant's response
        yield return StartCoroutine(GetResponse_precuedRef());

        // 💾 Log trial data, including stimulus parameters and participant response
        RegisterResponse_suprathres(RefLocation, Comp1Location, Comp2Location, TrialInfo, RGB_ref_string,
            RGB_comp_string, RGB_comp2_string, RGB_ref_corrected_string, RGB_comp_corrected_string,
            RGB_comp2_corrected_string, codedAnswer, validKeysString, pressedKeysString, actualWait_preStim,
            responseTime);

        // ➕ Move to the next trial
        trialCounter++;
    }

    /// <summary>
    /// Waits for exactly one valid response based on the precued Reference location,
    /// logs which physical direction was pressed (Top/Left/Right as a one-hot string),
    /// and sets `codedAnswer` to 1 if Comp1 was chosen or 2 if Comp2 was chosen.
    /// 
    /// Conventions (1-based positions):
    ///   1 = Top, 2 = Left, 3 = Right
    /// 
    /// Prompt & valid choices depend on where the Reference (Ref) is:
    ///   Ref=Top(1)  → choices: Left(A), Right(B)  → prompt "< >"
    ///   Ref=Left(2) → choices: Top(A),  Right(B)  → prompt "^ >"
    ///   Ref=Right(3)→ choices: Left(A), Top(B)    → prompt "< ^"
    /// 
    /// Keyboard keys:    ResponseKeyLeft / ResponseKeyRight / ResponseKeyTop
    /// Gamepad buttons:  West (Left) / East (Right) / North (Top)
    /// </summary>
    private IEnumerator GetResponse_precuedRef()
    {
        // ---------- 1) Build *direction detectors* that unify keyboard + gamepad ----------
        // Each Func<bool> returns true *once* in the frame where the mapped input is pressed.
        // - PressedLeft:  ResponseKeyLeft OR gamepad West
        // - PressedRight: ResponseKeyRight OR gamepad East
        // - PressedTop:   ResponseKeyTop OR gamepad North
        Func<bool> PressedLeft = () => Input.GetKeyDown(ResponseKeyLeft) || (gamepad != null && gamepad.buttonWest.wasPressedThisFrame);
        Func<bool> PressedRight = () => Input.GetKeyDown(ResponseKeyRight) || (gamepad != null && gamepad.buttonEast.wasPressedThisFrame);
        Func<bool> PressedTop = () => Input.GetKeyDown(ResponseKeyTop) || (gamepad != null && gamepad.buttonNorth.wasPressedThisFrame);

        // ---------- 2) From RefLocation, decide which TWO directions are valid this trial ----------
        // We name them A and B (first/second option) so we can:
        //   (a) show the correct prompt string, and
        //   (b) later map the pressed option (A or B) back to an absolute screen position.
        string prompt;
        Func<bool> PressedA; // first valid choice this trial
        Func<bool> PressedB; // second valid choice this trial

        // 1=Top  -> choices: Left(A) / Right(B)  => show "< >"
        // 2=Left -> choices: Top(A)  / Right(B)  => show "^ >"
        // 3=Right-> choices: Left(A) / Top(B)    => show "< ^"
        switch (RefLocation)
        {
            case 1: prompt = "< · >"; PressedA = PressedLeft; PressedB = PressedRight; validKeysString = "0_1_1"; break;
            case 2: prompt = "· ^ >"; PressedA = PressedTop; PressedB = PressedRight; validKeysString = "1_0_1"; break;
            case 3: prompt = "< ^ ·"; PressedA = PressedLeft; PressedB = PressedTop; validKeysString = "1_1_0"; break;
            default:
                Debug.LogError($"Invalid RefLocation {RefLocation}. Expected 1..3 (Top/Left/Right).");
                yield break;
        }

        // Show the two valid choices for this trial (based on Ref position).
        displayText.text = prompt;

        // ---------- 3) (Optional) one-time screenshot of the response probe ----------
        if (takeScreenshotOfStimuli_suprathres && trialCounter == 1)
        {
            string filename = Path.Combine(saveDirectory, "responseprobe.png");
            ScreenCapture.CaptureScreenshot(filename);
            yield return new WaitForEndOfFrame();
        }

        // ---------- 4) Start reaction-time clock ----------
        float responseStartTime = Time.time;

        // ---------- 5) Wait for EXACTLY ONE of the two valid responses ----------
        // We store "which" as 1 (A) or 2 (B) and return immediately on the first valid press.
        int which = 0; // 1 = A, 2 = B
        yield return new WaitUntil(() =>
        {
            if (PressedA()) { which = 1; return true; }
            if (PressedB()) { which = 2; return true; }
            return false;
        });

        // RT
        responseTime = Time.time - responseStartTime;

        // ---------- 6) Log the *actual physical direction* pressed as one-hot "Top_Left_Right" ----------
        // We do NOT store “options A/B,” we store which direction was pressed on screen.
        // The mapping from (RefLocation, which) to (Top/Left/Right) is fixed by the table above.
        int top = 0, left = 0, right = 0;
        switch (RefLocation)
        {
            case 1: left = (which == 1) ? 1 : 0; right = (which == 2) ? 1 : 0; break; // A=Left, B=Right
            case 2: top = (which == 1) ? 1 : 0; right = (which == 2) ? 1 : 0; break; // A=Top,  B=Right
            case 3: left = (which == 1) ? 1 : 0; top = (which == 2) ? 1 : 0; break; // A=Left, B=Top
        }
        // Example encodings:
        //   "0_0_1" → Right/East
        //   "0_1_0" → Left/West
        //   "1_0_0" → Top/North
        pressedKeysString = $"{top}_{left}_{right}";

        // ---------- 7) Convert which(A/B) into an *absolute screen location* (1=Top, 2=Left, 3=Right) ----------
        // This resolves the response from "first or second option" into the concrete position pressed.
        int chosenLocation = RefLocation switch
        {
            1 => (which == 1) ? 2 : 3, // Ref Top: A=Left(2),  B=Right(3)
            2 => (which == 1) ? 1 : 3, // Ref Left: A=Top(1),  B=Right(3)
            3 => (which == 1) ? 2 : 1, // Ref Right: A=Left(2), B=Top(1)
            _ => 0
        };

        // Final code: 1 if Comp1 chosen, else 2 (Comp2)
        codedAnswer = (chosenLocation == Comp1Location) ? 1f : 2f;
    }

    /// <summary>
    /// Logs the participant's response and stores trial-related data.
    /// This function ensures that AEPsych receives the response immediately 
    /// to compute the next trial efficiently while also recording the data for analysis.
    /// trialCounter        : Current trial number (starting from 0)
    /// trialInfo           : Information about the trial (e.g., Trial_10_Cond_1)
    /// RefLocation         : Position of the reference stimulus (1 = top; 2: bottom left; 3: bottom right)
    /// Comp1Location       : Position of the comparison stimulus #1
    /// Comp2Location       : Position of the comparison stimulus #2
    /// pressedKeys         : String representation of the response keys pressed (e.g., "1_0_0", "0_1_0", "0_0_1")
    /// RGB_ref             : RGB values of the reference stimulus before correction
    /// RGB_comp1           : RGB values of the fixed comparison stimulus before correction
    /// RGB_comp2           : RGB values of the comparison stimulus that changes 
    /// RGB_ref_corrected   : Corrected RGB values of the reference stimulus
    /// RGB_comp1_corrected : Corrected RGB values of the fixed comparison stimulus
    /// RGB_comp2_corrected : Corrected RGB values of the varying comparison stimulus
    /// codedResp           : Binary-coded response (1 = Correct, 0 = Incorrect)
    /// RT                  : Reaction time (in seconds)
    /// actual_wait_preStim : The time between the disappearance of '+' and stimulus presentation
    /// LUT                 : look-up table for gamma correction
    private void RegisterResponse_suprathres(int RefLocation, int Comp1Location, int Comp2Location,
        string TrialInfo, string ref_RGB, string comp1_RGB, string comp2_RGB, string ref_RGB_corrected,
        string comp1_RGB_corrected, string comp2_RGB_corrected, float binaryResp, string validKeys,
        string pressedKeys, float actual_wait_preStim, float respTime)
    {
        // 💾 Log response immediately for real-time trial adaptation
        // This allows AEPsych to compute the next trial placement as soon as possible.
        fileDataExtractor.WriteResponseToFileComp2(initFilePath, TrialInfo, ref_RGB, comp1_RGB, comp2_RGB, codedAnswer);

        // 💾 Append trial data for long-term storage & analysis
        // This ensures the collected responses are saved in a structured CSV file.
        AppendTrialData_suprathres(
            trialCounter.ToString(),
            TrialInfo,
            RefLocation.ToString(),
            Comp1Location.ToString(),
            Comp2Location.ToString(),
            validKeys,
            pressedKeys,
            ref_RGB,
            comp1_RGB,
            comp2_RGB,
            ref_RGB_corrected,
            comp1_RGB_corrected,
            comp2_RGB_corrected,
            binaryResp.ToString(),
            actual_wait_preStim.ToString(),
            respTime.ToString()
        );
    }

    private void createTrialData_suprathres()
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
                sw.WriteLine("TrialCounter, TrialInfo, RefLocation, Comp1Location, Comp2Location, " +
                "ValidKeys(top_left_right), PressedKey(top_left_right), Ref, Comp1, Comp2, Ref_Corrected, " +
                "Comp1_Corrected, Comp2_Corrected, binaryResp, actual_wait_preStim, RT, " +
                $"subID_{subject_id.ToString()}, subInit_{subject_init}, sessNum_{session_today.ToString()}, " +
                $"seed_{subject_seed.ToString()}, LUT_{gammaCorrectionFile_suprathres}");
            }
        }
    }

    private void AppendTrialData_suprathres(
        string trialCounter, string trialInfo, string RefLocation, string Comp1Location,
        string Comp2Location, string validKeys, string pressedKeys, string RGB_ref, string RGB_comp1,
        string RGB_comp2, string RGB_ref_corrected, string RGB_comp1_corrected, string RGB_comp2_corrected,
        string codedResp, string actual_wait_preStim, string RT)
    {
        using (StreamWriter sw = new StreamWriter(csvFilePath, true))
        {
            // Write to CSV file with better formatting (split into two lines for readability)
            sw.WriteLine(
                $"{trialCounter}, {trialInfo}, {RefLocation}, {Comp1Location}, {Comp2Location}, {validKeys}, " +
                $"{pressedKeys}, {RGB_ref}, {RGB_comp1}, {RGB_comp2}, {RGB_ref_corrected}, {RGB_comp1_corrected}, " +
                $"{RGB_comp2_corrected}, {codedResp}, {actual_wait_preStim}, {RT}"
            );
        }
    }

}



