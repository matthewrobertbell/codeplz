use std::cmp::min;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{anyhow, bail, Context};
use aws_sdk_bedrock::config;
use axum::extract::State;
use globset::{Glob, GlobSetBuilder};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
use similar::{ChangeTag, TextDiff};
use tiktoken_rs::cl100k_base;
use walkdir::WalkDir;

use askama::Template;
use axum::{
    routing::{get, post},
    Json, Router,
};

#[derive(Serialize, Deserialize, Debug, Default, Clone)]
enum LLMModel {
    #[default]
    OpenAIGPT4o,
    Claude35SonnetBedrock,
}

impl LLMModel {
    fn max_tokens(&self) -> usize {
        match self {
            LLMModel::OpenAIGPT4o => 100_000,
            LLMModel::Claude35SonnetBedrock => 180_000,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Default, Clone)]
struct Config {
    include_paths: Vec<String>,
    exclude_paths: Vec<String>,
    commands: Vec<String>,
    #[serde(default)]
    use_gitignore: bool,
    system_prompt: Option<String>,
    model: LLMModel,
    maximum_context_tokens: Option<usize>,
}

#[derive(Serialize, Deserialize, Debug)]
struct OpenAIRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
}

#[derive(Serialize, Deserialize, Debug)]
struct OpenAIMessage {
    role: String,
    content: String,
}

#[derive(Deserialize, Debug)]
struct OpenAIResponse {
    choices: Vec<OpenAIChoice>,
}

#[derive(Deserialize, Debug)]
struct OpenAIChoice {
    message: OpenAIMessage,
}

#[derive(Debug, Deserialize)]
struct LLMResponse {
    explanation: String,
    changes: Vec<Change>,
    conclusion: String,
}

#[derive(Debug, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
enum LineOrLines {
    Line(String),
    Lines(Vec<String>),
}

impl LineOrLines {
    fn lines(&self) -> Vec<String> {
        match self {
            LineOrLines::Line(line) => vec![line.clone()],
            LineOrLines::Lines(lines) => lines.clone(),
        }
    }

    fn len(&self) -> usize {
        match self {
            LineOrLines::Line(_) => 1,
            LineOrLines::Lines(lines) => lines.len(),
        }
    }
}

#[derive(Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "command", rename_all = "SCREAMING_SNAKE_CASE")]
enum LLMCommand {
    InsertAfter {
        insert_lines: LineOrLines,
        marker_lines: LineOrLines,
    },
    InsertBefore {
        insert_lines: LineOrLines,
        marker_lines: LineOrLines,
    },
    Delete {
        delete_lines: LineOrLines,
    },
    CreateFile {
        new_lines: LineOrLines,
    },
    RenameFile {
        new_filename: PathBuf,
    },
    DeleteFile,
}

#[derive(Debug, Deserialize, Serialize)]
struct Change {
    filename: PathBuf,
    #[serde(flatten)]
    command: LLMCommand,
    reason: String,
}

#[derive(Debug, Serialize)]
struct ChangeWithDiff {
    #[serde(flatten)]
    change: Change,
    diff: String,
}

async fn zzz() -> anyhow::Result<()> {
    /*
    // Interactive choices for user for stuff that can't be inferred like model?

    // Step 2: mkcontext functionality, build up the prompt
    let maximum_context_tokens = config
        .maximum_context_tokens
        .unwrap_or_else(|| config.model.max_tokens());
    let context = make_context(maximum_context_tokens, &config)?;

    // Step 3: get response from LLM
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <prompt>", args[0]);
        std::process::exit(1);
    }

    let prompt = args[1..].join(" ").trim_matches('"').to_string();
    let prompt = format!("{} {}", context, prompt);

    let system_prompt = config
        .system_prompt
        .unwrap_or_else(|| include_str!("default_system_prompt.txt").to_string());

    let llm_response = match config.model {
        LLMModel::Claude35SonnetBedrock => {
            let aws_config = aws_config::load_from_env().await;
            let bedrock_client = aws_sdk_bedrock::Client::new(&aws_config);

            "placeholder".to_string()
        }
        LLMModel::OpenAIGPT4o => {
            let api_key = std::env::var("OPENAI_API_KEY")
                .context("OPENAI_API_KEY environment variable not set")?;

            let client = Client::new();
            let request = OpenAIRequest {
                model: "gpt-4o".to_string(),
                messages: vec![
                    OpenAIMessage {
                        role: "system".to_string(),
                        content: system_prompt,
                    },
                    OpenAIMessage {
                        role: "user".to_string(),
                        content: prompt,
                    },
                ],
            };

            let response = client
                .post("https://api.openai.com/v1/chat/completions")
                .header("Authorization", format!("Bearer {}", api_key))
                .json(&request)
                .send()
                .await
                .context("Failed to send request to OpenAI API")?;

            let openai_response: OpenAIResponse = response
                .json()
                .await
                .context("Failed to parse OpenAI API response")?;

            openai_response
                .choices
                .first()
                .context("No response from OpenAI API")?
                .message
                .content
                .clone()
        }
    };

    let llm_response = llm_response.replace("```json", "").replace("```", "");
    println!("LLM Response: {}", llm_response);

    // Step 4: Modify files based on LLM response

    let response: LLMResponse =
        serde_json::from_str(&llm_response).context("Failed to parse JSON content")?;

    println!("{}\n------", response.explanation);

    for change in &response.changes {
        if !is_file_in_current_directory(&change.filename) {
            println!(
                "Warning: Filename '{}' is outside the current directory. Skipping.",
                change.filename.display()
            );
            continue;
        }

        println!();

        println!("=> File: {}", change.filename.display());
        println!(
            "=> Action: {}",
            match change.command {
                LLMCommand::InsertBefore { .. } => "Inserting new lines before a marker",
                LLMCommand::InsertAfter { .. } => "Inserting new lines after a marker",
                LLMCommand::Delete { .. } => "Deleting lines",
                LLMCommand::CreateFile { .. } => "Creating a new file",
                LLMCommand::RenameFile { .. } => "Renaming a file",
                LLMCommand::DeleteFile => "Deleting a file",
            }
        );
        println!("=> Reason: {}", change.reason);

        match &change.command {
            LLMCommand::CreateFile { new_lines } => {
                let file_path = Path::new(&change.filename);
                if file_path.exists() {
                    bail!("File already exists: {:?}", change.filename);
                }
                if let Some(parent) = file_path.parent() {
                    fs::create_dir_all(parent)
                        .with_context(|| format!("✗ Failed to create directory: {:?}", parent))?;
                }
                fs::write(file_path, new_lines.lines().join("\n")).with_context(|| {
                    format!("✗ Failed to create file: {}", change.filename.display())
                })?;
                println!(
                    "✓ Created file {} and inserted {} lines",
                    change.filename.display(),
                    new_lines.len()
                );
            }
            LLMCommand::RenameFile { new_filename } => {
                fs::rename(&change.filename, new_filename).with_context(|| {
                    format!("✗ Failed to rename file: {}", change.filename.display())
                })?;
                println!(
                    "✓ Renamed file: {} -> {}",
                    change.filename.display(),
                    new_filename.display()
                );
            }
            LLMCommand::DeleteFile => {
                fs::remove_file(&change.filename)
                    .with_context(|| format!("✗ Failed to delete file: {:?}", change.filename))?;
                println!("✓ Deleted file: {:?}", change.filename);
            }
            LLMCommand::InsertBefore {
                insert_lines,
                marker_lines,
            } => {
                let file_lines = fs::read_to_string(&change.filename)
                    .with_context(|| format!("✗ Failed to read file: {:?}", change.filename))?
                    .lines()
                    .map(String::from)
                    .collect::<Vec<_>>();

                if let Some(index) = find_in_file_lines(&file_lines, &marker_lines.lines()) {
                    let mut insert_lines = insert_lines.lines();
                    let marker_lines = marker_lines.lines();

                    // Remove marker lines from insert_lines if they match
                    if insert_lines.len() >= marker_lines.len()
                        && insert_lines
                            .iter()
                            .take(marker_lines.len())
                            .map(|s| s.trim())
                            .eq(marker_lines.iter().map(|s| s.trim()))
                    {
                        insert_lines = insert_lines.into_iter().skip(marker_lines.len()).collect();
                    }

                    let mut new_lines = file_lines[..index].to_vec();
                    new_lines.extend(insert_lines.clone());
                    new_lines.extend(file_lines[index..].iter().cloned());
                    fs::write(&change.filename, new_lines.join("\n")).with_context(|| {
                        format!("✗ Failed to write to file: {:?}", change.filename)
                    })?;
                    println!(
                        "✓ Inserted {} lines into {}",
                        insert_lines.len(),
                        change.filename.display()
                    );
                } else {
                    bail!(
                        "Failed to find {} lines in {:?}",
                        marker_lines.len(),
                        change.filename.display()
                    );
                }
            }
            LLMCommand::InsertAfter {
                marker_lines,
                insert_lines,
            } => {
                let file_lines = fs::read_to_string(&change.filename)
                    .with_context(|| format!("✗ Failed to read file: {:?}", change.filename))?
                    .lines()
                    .map(String::from)
                    .collect::<Vec<_>>();

                if marker_lines.len() == 0 && file_lines.is_empty() {
                    // This is the start of a new file
                    fs::write(&change.filename, insert_lines.lines().join("\n")).with_context(
                        || format!("✗ Failed to write to file: {:?}", change.filename),
                    )?;
                    println!(
                        "✓ Inserted {} lines into {}",
                        insert_lines.len(),
                        change.filename.display()
                    );
                } else if let Some(index) = find_in_file_lines(&file_lines, &marker_lines.lines()) {
                    let mut insert_lines = insert_lines.lines();
                    let marker_lines = marker_lines.lines();

                    // Remove marker lines from insert_lines if they match
                    if insert_lines.len() >= marker_lines.len()
                        && insert_lines
                            .iter()
                            .take(marker_lines.len())
                            .map(|s| s.trim())
                            .eq(marker_lines.iter().map(|s| s.trim()))
                    {
                        insert_lines = insert_lines.into_iter().skip(marker_lines.len()).collect();
                    }

                    let mut new_lines = file_lines[..=index + marker_lines.len() - 1].to_vec();
                    new_lines.extend(insert_lines.clone());
                    new_lines.extend(file_lines[index + marker_lines.len()..].iter().cloned());
                    fs::write(&change.filename, new_lines.join("\n")).with_context(|| {
                        format!("✗ Failed to write to file: {:?}", change.filename)
                    })?;
                    println!(
                        "✓ Inserted {} lines into {}",
                        insert_lines.len(),
                        change.filename.display()
                    );
                } else {
                    bail!(
                        "Failed to find {} lines in {:?}",
                        marker_lines.len(),
                        change.filename.display()
                    );
                }
            }
            LLMCommand::Delete { delete_lines } => {
                let file_lines = fs::read_to_string(&change.filename)
                    .with_context(|| format!("✗ Failed to read file: {:?}", change.filename))?
                    .lines()
                    .map(String::from)
                    .collect::<Vec<_>>();

                dbg!(&delete_lines.lines());

                if let Some(start_index) = find_in_file_lines(&file_lines, &delete_lines.lines()) {
                    let mut new_lines = file_lines[..start_index].to_vec();
                    new_lines.extend(
                        file_lines[start_index + delete_lines.lines().len()..]
                            .iter()
                            .cloned(),
                    );
                    fs::write(&change.filename, new_lines.join("\n")).with_context(|| {
                        format!("✗ Failed to write to file: {:?}", change.filename)
                    })?;
                    println!(
                        "✓ Deleted {} lines in {:?}",
                        delete_lines.len(),
                        change.filename.display()
                    );
                } else {
                    bail!(
                        "Failed to find {} lines to delete in {:?}",
                        delete_lines.len(),
                        change.filename.display()
                    );
                }
            }
        }
    }

    if !response.changes.is_empty() {
        println!("------");
    }

    println!(" {}", response.conclusion);

    */

    Ok(())
}

#[derive(Clone)]
struct AppState {
    config: Config,
}

impl AppState {
    fn new(config: Config) -> Self {
        AppState { config }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config_path = "codeplz.json";
    let config = match std::fs::read_to_string(config_path) {
        Ok(content) => serde_json::from_str::<Config>(&content)?,
        Err(_) => {
            let mut default_config = Config::default();
            default_config
                .exclude_paths
                .push("codeplz.json".to_string());
            let config_json = serde_json::to_string_pretty(&default_config).unwrap();
            std::fs::write(config_path, config_json).context("Failed to create config file")?;
            default_config
        }
    };
    println!("Loaded config: {:?}", config);

    // Create AppState with the loaded config
    let state = AppState::new(config);

    // Update the main function to include the new route
    let app = Router::new()
        .route("/", get(index))
        .route("/prompt", post(prompt))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000").await?;
    println!("Server running on http://127.0.0.1:3000");
    axum::serve(listener, app).await?;

    Ok(())
}

#[derive(Template)]
#[template(path = "index.html")]
struct IndexTemplate {
    code_path: String,
    model_name: String,
}

async fn index(State(state): State<AppState>) -> IndexTemplate {
    let code_path = std::env::current_dir()
        .map(|path| path.to_string_lossy().into_owned())
        .unwrap_or_else(|_| String::from("Unknown"));
    let model = match state.config.model {
        LLMModel::OpenAIGPT4o => "OpenAI GPT-4",
        LLMModel::Claude35SonnetBedrock => "Claude 3.5 Sonnet (Bedrock)",
    };
    IndexTemplate {
        code_path,
        model_name: model.to_string(),
    }
}

// Add this new struct
#[derive(Deserialize)]
struct PromptRequest {
    prompt: String,
}

// Add this new struct
#[derive(Serialize)]
struct PromptResponse {
    explanation: String,
    changes: Vec<ChangeWithDiff>,
    conclusion: String,
    token_count: usize,
}

// Replace the existing prompt function with this new implementation
async fn prompt(
    State(state): State<AppState>,
    Json(request): Json<PromptRequest>,
) -> Json<PromptResponse> {
    let config = &state.config;
    let maximum_context_tokens = config
        .maximum_context_tokens
        .unwrap_or_else(|| config.model.max_tokens());

    // Generate context
    let context = make_context(maximum_context_tokens, config).unwrap_or_else(|e| {
        eprintln!("Error generating context: {}", e);
        String::new()
    });

    // Combine context and user prompt
    let full_prompt = format!("{}\n\nUser request: {}", context, request.prompt);

    // Call LLM based on the configured model
    let llm_response = match config.model {
        LLMModel::OpenAIGPT4o => call_openai_gpt4(&full_prompt).await,
        LLMModel::Claude35SonnetBedrock => call_claude_bedrock(&full_prompt).await,
    }
    .replace("```json", "")
    .replace("```", "");

    dbg!(&llm_response);

    // Parse LLM response
    let llm_data: LLMResponse = match serde_json::from_str(&llm_response) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Error parsing LLM response: {}", e);
            return Json(PromptResponse {
                explanation: "Error processing request".to_string(),
                changes: vec![],
                conclusion: "An error occurred".to_string(),
                token_count: 0,
            });
        }
    };

    // Validate changes
    let validated_changes = validate_changes(llm_data.changes);

    // Generate diffs for each validated change
    let changes_with_diff = validated_changes
        .into_iter()
        .map(|change| {
            let diff = generate_diff(&change, &state.config)
                .unwrap_or_else(|_| String::from("Unable to generate diff"));
            ChangeWithDiff { change, diff }
        })
        .collect::<Vec<ChangeWithDiff>>();

    // Calculate token count (you may want to implement a more accurate method)
    let token_count = llm_response.len() / 4; // Rough estimate

    Json(PromptResponse {
        explanation: llm_data.explanation,
        changes: changes_with_diff,
        conclusion: llm_data.conclusion,
        token_count,
    })
}

fn validate_changes(changes: Vec<Change>) -> Vec<Change> {
    changes
        .into_iter()
        .filter_map(|change| match validate_change(&change) {
            Ok(_) => Some(change),
            Err(e) => {
                eprintln!("Failed to validate change: {:?}", e);
                None
            }
        })
        .collect()
}

fn validate_change(change: &Change) -> anyhow::Result<()> {
    match &change.command {
        LLMCommand::CreateFile { .. } => {
            // Check if the file already exists
            if change.filename.exists() {
                bail!("File already exists: {:?}", change.filename);
            }
        }
        LLMCommand::RenameFile { new_filename } => {
            // Check if the source file exists and the destination doesn't
            if !change.filename.exists() {
                bail!("Source file does not exist: {:?}", change.filename);
            }
            if new_filename.exists() {
                bail!("Destination file already exists: {:?}", new_filename);
            }
        }
        LLMCommand::DeleteFile => {
            // Check if the file exists
            if !change.filename.exists() {
                bail!("File does not exist: {:?}", change.filename);
            }
        }
        LLMCommand::InsertAfter { marker_lines, .. }
        | LLMCommand::InsertBefore { marker_lines, .. }
        | LLMCommand::Delete {
            delete_lines: marker_lines,
        } => {
            // Check if the file exists and the marker lines can be found
            let file_content = std::fs::read_to_string(&change.filename)?;
            let file_lines: Vec<String> = file_content.lines().map(String::from).collect();
            if find_in_file_lines(&file_lines, &marker_lines.lines()).is_none() {
                bail!("Failed to find marker lines in file: {:?}", change.filename);
            }
        }
    }
    Ok(())
}

fn generate_diff(change: &Change, config: &Config) -> anyhow::Result<String> {
    match &change.command {
        LLMCommand::CreateFile { new_lines } => {
            // For new files, show all lines as added
            Ok(new_lines
                .lines()
                .into_iter()
                .map(|line| format!("+{}\n", line))
                .collect())
        }
        _ => {
            let file_path = &change.filename;
            let old_content = std::fs::read_to_string(file_path)?;
            let new_content = apply_change_to_content(&old_content, change)?;

            let diff = TextDiff::from_lines(&old_content, &new_content);
            let mut diff_output = String::new();

            for change in diff.iter_all_changes() {
                let sign = match change.tag() {
                    ChangeTag::Delete => "-",
                    ChangeTag::Insert => "+",
                    ChangeTag::Equal => " ",
                };
                diff_output.push_str(&format!("{}{}", sign, change));
            }

            Ok(diff_output)
        }
    }
}

fn apply_change_to_content(content: &str, change: &Change) -> anyhow::Result<String> {
    let lines: Vec<String> = content.lines().map(String::from).collect();
    let mut new_lines = lines.clone();

    match &change.command {
        LLMCommand::CreateFile {
            new_lines: create_lines,
        } => {
            // For new files, return the new content directly
            return Ok(create_lines.lines().join("\n"));
        }
        LLMCommand::InsertAfter {
            marker_lines,
            insert_lines,
        } => {
            let marker_lines: Vec<String> =
                marker_lines.lines().into_iter().map(String::from).collect();
            if let Some(index) = find_in_file_lines(&lines, &marker_lines) {
                new_lines.splice(
                    index + marker_lines.len()..index + marker_lines.len(),
                    insert_lines.lines().into_iter().map(String::from),
                );
            }
        }
        LLMCommand::InsertBefore {
            marker_lines,
            insert_lines,
        } => {
            let marker_lines: Vec<String> =
                marker_lines.lines().into_iter().map(String::from).collect();
            if let Some(index) = find_in_file_lines(&lines, &marker_lines) {
                new_lines.splice(
                    index..index,
                    insert_lines.lines().into_iter().map(String::from),
                );
            }
        }
        LLMCommand::Delete { delete_lines } => {
            if let Some(index) = find_in_file_lines(&lines, &delete_lines.lines()) {
                new_lines.drain(index..index + delete_lines.lines().len());
            }
        }
        LLMCommand::RenameFile { .. } | LLMCommand::DeleteFile => {}
    }

    Ok(new_lines.join("\n"))
}

async fn call_openai_gpt4(prompt: &str) -> String {
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    let client = reqwest::Client::new();
    let response = client
        .post("https://api.openai.com/v1/chat/completions")
        .header("Authorization", format!("Bearer {}", api_key))
        .json(&json!({
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": include_str!("default_system_prompt.txt")},
                {"role": "user", "content": prompt}
            ]
        }))
        .send()
        .await
        .expect("Failed to send request to OpenAI API");

    let response_json: serde_json::Value = response
        .json()
        .await
        .expect("Failed to parse OpenAI API response");
    response_json["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string()
}

async fn call_claude_bedrock(prompt: &str) -> String {
    /*
    let aws_config = aws_config::load_from_env().await;
    let bedrock_client = aws_sdk_bedrock::Client::new(&aws_config);

    let request = bedrock_client
        .invoke_model()
        .model_id("anthropic.claude-v2")
        .body(
            json!({
                "prompt": format!("Human: {}\n\nAssistant:", prompt),
                "max_tokens_to_sample": 8000,
                "temperature": 0.5,
                "top_p": 1,
            })
            .to_string(),
        );

    let response = request.send().await.expect("Failed to call Claude Bedrock");
    let response_body = response
        .body()
        .as_ref()
        .expect("Empty response from Claude Bedrock");
    let response_json: serde_json::Value =
        serde_json::from_slice(response_body).expect("Failed to parse Claude Bedrock response");
    response_json["completion"]
        .as_str()
        .unwrap_or("")
        .to_string()
         */
    "placeholder".to_string()
}

fn is_file_in_current_directory(path: &Path) -> bool {
    path.is_relative() && !path.starts_with("..")
}

fn levenshtein_distance(s1: &str, s2: &str) -> usize {
    let len1 = s1.chars().count();
    let len2 = s2.chars().count();
    let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];

    for (i, row) in matrix.iter_mut().enumerate() {
        row[0] = i;
    }
    for j in 0..=len2 {
        matrix[0][j] = j;
    }

    for (i, c1) in s1.chars().enumerate() {
        for (j, c2) in s2.chars().enumerate() {
            let cost = if c1 == c2 { 0 } else { 1 };
            matrix[i + 1][j + 1] = min(
                min(matrix[i][j + 1] + 1, matrix[i + 1][j] + 1),
                matrix[i][j] + cost,
            );
        }
    }

    matrix[len1][len2]
}

fn find_in_file_lines(file_lines: &[String], needle: &[String]) -> Option<usize> {
    let non_empty_needle: Vec<_> = needle
        .iter()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();

    if non_empty_needle.is_empty() {
        return Some(0);
    }

    let needle_joined = non_empty_needle.join("\n");
    let needle_len = needle_joined.chars().count();
    let mut best_match = None;
    let mut min_distance = usize::MAX;

    for (i, window) in file_lines.windows(needle.len()).enumerate() {
        let window_joined = window
            .iter()
            .map(|s| s.trim())
            .collect::<Vec<_>>()
            .join("\n");
        let distance = levenshtein_distance(&needle_joined, &window_joined);

        if distance < min_distance {
            min_distance = distance;
            best_match = Some(i);
        }

        if distance == 0 {
            break; // Exact match found
        }
    }

    // Check if the best match meets the 95% similarity threshold
    if let Some(i) = best_match {
        let similarity = 1.0 - (min_distance as f64 / needle_len as f64);

        if similarity >= 0.95 {
            return Some(i);
        } else {
            println!("Best match similarity: {}", similarity);
            println!(
                "Best match: {:?}",
                &file_lines[i..min(i + needle.len(), file_lines.len())]
            );
        }
    }

    None
}

fn make_context(maximum_context_tokens: usize, config: &Config) -> anyhow::Result<String> {
    let mut content = String::new();
    let mut current_token_count = 0;
    let bpe = cl100k_base().unwrap();

    let mut include_builder = GlobSetBuilder::new();
    let mut exclude_builder = GlobSetBuilder::new();

    for glob in &config.include_paths {
        include_builder
            .add(Glob::new(glob).map_err(|e| anyhow!("Invalid include glob pattern: {}", e))?);
    }

    for glob in &config.exclude_paths {
        exclude_builder
            .add(Glob::new(glob).map_err(|e| anyhow!("Invalid exclude glob pattern: {}", e))?);
    }

    if config.include_paths.is_empty() {
        include_builder.add(Glob::new("**/*").unwrap());
    }

    let include_set = include_builder
        .build()
        .map_err(|e| anyhow!("Failed to build include globset: {}", e))?;
    let exclude_set = exclude_builder
        .build()
        .map_err(|e| anyhow!("Failed to build exclude globset: {}", e))?;

    for entry in WalkDir::new(".") {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            continue;
        }

        let path_str = path.to_str().ok_or_else(|| anyhow!("Invalid path"))?;
        let path_str = path_str.strip_prefix("./").unwrap_or(path_str);

        if (include_set.is_match(path_str)) && !exclude_set.is_match(path_str) {
            let file_content = match fs::read_to_string(path) {
                Ok(content) => content,
                Err(e) => {
                    if e.kind() == std::io::ErrorKind::InvalidData {
                        eprintln!("Ignored non-UTF8 file: {path_str}");
                    } else {
                        eprintln!("Failed to read file: {path_str}");
                    }
                    continue;
                }
            };

            let file_context = format!(
                r#"File name: "{}"

File contents: """
{}"""
----------

"#,
                path_str, file_content
            );
            let new_token_count = add_content(
                &mut content,
                current_token_count,
                &file_context,
                &bpe,
                maximum_context_tokens,
            )?;
            println!("Processed file: {path_str:<70} - {new_token_count:>6} tokens");
            current_token_count += new_token_count;
        }
    }

    // Process commands
    for cmd in &config.commands {
        let output = execute_command(cmd)?;
        let command_context = format!(
            r#"Command: "{}"

Command output: """
{}"""
----------

"#,
            cmd, output
        );
        let new_token_count = add_content(
            &mut content,
            current_token_count,
            &command_context,
            &bpe,
            maximum_context_tokens,
        )?;
        println!("Executed command: {cmd:<67}  - {new_token_count:>6} tokens");
        current_token_count += new_token_count;
    }

    let content = content.replace('\n', "\r\n");

    println!(
        "Success loading context! Token count: {}",
        current_token_count
    );
    Ok(content)
}

fn add_content(
    content: &mut String,
    current_token_count: usize,
    new_content: &str,
    bpe: &tiktoken_rs::CoreBPE,
    token_limit: usize,
) -> anyhow::Result<usize> {
    let new_token_count = bpe.encode_with_special_tokens(new_content).len();

    if current_token_count + new_token_count > token_limit {
        return Err(anyhow::anyhow!("Token limit exceeded"));
    }
    if new_token_count > 0 {
        content.push_str(new_content);
    }
    Ok(new_token_count)
}

fn execute_command(cmd: &str) -> anyhow::Result<String> {
    let output = Command::new("sh")
        .arg("-c")
        .arg(cmd)
        .output()
        .with_context(|| format!("Failed to execute command: {}", cmd))?;

    let exit_status = output.status.code().unwrap_or(-1);
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();

    Ok(format!(
        "Exit Status: {}\n\nSTDOUT:\n{}\n\nSTDERR:\n{}",
        exit_status, stdout, stderr
    ))
}
