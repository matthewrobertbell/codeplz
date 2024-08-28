use std::cmp::min;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{anyhow, bail, Context};
use askama::Template;
use axum::extract::State;
use axum::response::IntoResponse;
use axum::{
    routing::{get, post},
    Json, Router,
};
use globset::{Glob, GlobSetBuilder};
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use similar::{ChangeTag, TextDiff};
use tiktoken_rs::cl100k_base;
use walkdir::WalkDir;

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
        .route("/apply_changes", post(apply_changes))
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

#[derive(Serialize)]
struct ProcessedFile {
    name: String,
    token_count: usize,
}

#[derive(Serialize)]
struct PromptResponse {
    explanation: String,
    changes: Vec<ChangeWithDiff>,
    conclusion: String,
    input_token_count: usize,
    output_token_count: usize,
    processed_files: Vec<ProcessedFile>,
}

// Replace the existing prompt function with this new implementation
async fn prompt(
    State(state): State<AppState>,
    Json(request): Json<PromptRequest>,
) -> Result<Json<PromptResponse>, (StatusCode, String)> {
    let config = &state.config;
    let maximum_context_tokens = config
        .maximum_context_tokens
        .unwrap_or_else(|| config.model.max_tokens());

    // Get the system prompt
    let system_prompt = config
        .system_prompt
        .clone()
        .unwrap_or_else(|| include_str!("default_system_prompt.txt").to_string());

    // Generate context and track processed files
    let (context, processed_files) =
        make_context(maximum_context_tokens, config).unwrap_or_else(|e| {
            eprintln!("Error generating context: {}", e);
            (String::new(), Vec::new())
        });

    // Combine system prompt, context, and user prompt
    let full_prompt = format!(
        "{}\n\n{}\n\nUser request: {}",
        system_prompt, context, request.prompt
    );

    // Calculate input token count
    let bpe = cl100k_base().unwrap();
    let input_token_count = bpe.encode_with_special_tokens(&full_prompt).len();

    // Call LLM based on the configured model
    let llm_response = match config.model {
        LLMModel::OpenAIGPT4o => call_openai_gpt4(&system_prompt, &full_prompt).await?,
        LLMModel::Claude35SonnetBedrock => call_claude_bedrock(&full_prompt).await,
    }
    .replace("```json", "")
    .replace("```", "");

    // Calculate output token count
    let output_token_count = bpe.encode_with_special_tokens(&llm_response).len();

    dbg!(&llm_response);

    // Parse LLM response
    let llm_data: LLMResponse = serde_json::from_str(&llm_response).map_err(|e| {
        eprintln!("Error parsing LLM response: {}", e);
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Error processing request".to_string(),
        )
    })?;

    // Generate diffs for each validated change
    let changes_with_diff = validate_changes(llm_data.changes)
        .into_iter()
        .map(|change| {
            let diff =
                generate_diff(&change).unwrap_or_else(|_| String::from("Unable to generate diff"));
            ChangeWithDiff { change, diff }
        })
        .collect::<Vec<ChangeWithDiff>>();

    let response = PromptResponse {
        explanation: llm_data.explanation,
        changes: changes_with_diff,
        conclusion: llm_data.conclusion,
        input_token_count,
        output_token_count,
        processed_files,
    };

    Ok(Json(response))
}

async fn call_openai_gpt4(
    system_prompt: &str,
    full_prompt: &str,
) -> Result<String, (StatusCode, String)> {
    let api_key = std::env::var("OPENAI_API_KEY").map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "OPENAI_API_KEY environment variable not set".to_string(),
        )
    })?;

    let client = Client::new();
    let request = OpenAIRequest {
        model: "gpt-4o-2024-08-06".to_string(),
        messages: vec![
            OpenAIMessage {
                role: "system".to_string(),
                content: system_prompt.to_string(),
            },
            OpenAIMessage {
                role: "user".to_string(),
                content: full_prompt.to_string(),
            },
        ],
    };

    let response = client
        .post("https://api.openai.com/v1/chat/completions")
        .header("Authorization", format!("Bearer {}", api_key))
        .json(&request)
        .send()
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to send request to OpenAI API: {}", e),
            )
        })?;

    let response_text = response.text().await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to read response from OpenAI API: {}", e),
        )
    })?;

    let openai_response: OpenAIResponse = serde_json::from_str(&response_text).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to parse OpenAI API response: {}", e),
        )
    })?;

    openai_response
        .choices
        .first()
        .map(|choice| choice.message.content.clone())
        .ok_or((
            StatusCode::INTERNAL_SERVER_ERROR,
            "No response from OpenAI API".to_string(),
        ))
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
    // First, check if the file is in the current directory
    if !is_file_in_current_directory(&change.filename) {
        bail!(
            "File is outside the current directory: {:?}",
            change.filename
        );
    }

    match &change.command {
        LLMCommand::CreateFile { .. } => {
            // Remove the check for existing files
        }
        LLMCommand::RenameFile { new_filename } => {
            // Check if the source file exists
            if !change.filename.exists() {
                bail!("Source file does not exist: {:?}", change.filename);
            }
            // Also check if the new filename is in the current directory
            if !is_file_in_current_directory(new_filename) {
                bail!(
                    "New filename is outside the current directory: {:?}",
                    new_filename
                );
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

fn generate_diff(change: &Change) -> anyhow::Result<String> {
    match &change.command {
        LLMCommand::CreateFile { new_lines } => {
            // For new files, show all lines as added
            Ok(new_lines
                .lines()
                .into_iter()
                .map(|line| format!("+{}\n", line))
                .fold(String::new(), |acc, line| acc + &line))
        }
        _ => {
            let file_path = &change.filename;
            let old_content = std::fs::read_to_string(file_path)?;
            let new_content = apply_change_to_content(&old_content, change)?;

            let diff = TextDiff::from_lines(&old_content, &new_content);
            let mut diff_output = String::new();
            let mut unchanged_lines = Vec::new();

            for change in diff.iter_all_changes() {
                match change.tag() {
                    ChangeTag::Equal => {
                        unchanged_lines.push(change.to_string());
                        if unchanged_lines.len() > 10 {
                            unchanged_lines.remove(0);
                        }
                    }
                    ChangeTag::Delete | ChangeTag::Insert => {
                        // Output up to 5 unchanged lines before the change
                        let start = unchanged_lines.len().saturating_sub(5);
                        for line in &unchanged_lines[start..] {
                            diff_output.push_str(&format!(" {}", line));
                        }
                        unchanged_lines.clear();

                        // Output the changed line
                        let sign = if change.tag() == ChangeTag::Delete {
                            "-"
                        } else {
                            "+"
                        };
                        diff_output.push_str(&format!("{}{}", sign, change));
                    }
                }
            }

            // Output up to 5 unchanged lines after the last change
            let end = unchanged_lines.len().min(5);
            for line in &unchanged_lines[..end] {
                diff_output.push_str(&format!(" {}", line));
            }

            Ok(diff_output)
        }
    }
}

#[derive(Deserialize)]
struct ApplyChangesRequest {
    changes: Vec<Change>,
}

#[derive(Serialize)]
struct ApplyChangesResponse {
    results: Vec<ChangeResult>,
}

#[derive(Serialize)]
struct ChangeResult {
    filename: String,
    success: bool,
    message: String,
}

async fn apply_changes(Json(request): Json<ApplyChangesRequest>) -> impl IntoResponse {
    let mut results = Vec::new();

    for change in request.changes {
        let result = apply_change(&change);
        results.push(result);
    }

    Json(ApplyChangesResponse { results })
}

fn apply_change(change: &Change) -> ChangeResult {
    match &change.command {
        LLMCommand::CreateFile { new_lines } => {
            let file_path = Path::new(&change.filename);
            if let Some(parent) = file_path.parent() {
                if let Err(e) = fs::create_dir_all(parent) {
                    return ChangeResult {
                        filename: change.filename.to_string_lossy().into_owned(),
                        success: false,
                        message: format!("Failed to create directory: {}", e),
                    };
                }
            }
            match fs::write(file_path, new_lines.lines().join("\n")) {
                Ok(_) => ChangeResult {
                    filename: change.filename.to_string_lossy().into_owned(),
                    success: true,
                    message: format!("Created file and inserted {} lines", new_lines.len()),
                },
                Err(e) => ChangeResult {
                    filename: change.filename.to_string_lossy().into_owned(),
                    success: false,
                    message: format!("Failed to create file: {}", e),
                },
            }
        }
        LLMCommand::RenameFile { new_filename } => {
            match fs::rename(&change.filename, new_filename) {
                Ok(_) => ChangeResult {
                    filename: change.filename.to_string_lossy().into_owned(),
                    success: true,
                    message: format!("Renamed file to {}", new_filename.to_string_lossy()),
                },
                Err(e) => ChangeResult {
                    filename: change.filename.to_string_lossy().into_owned(),
                    success: false,
                    message: format!("Failed to rename file: {}", e),
                },
            }
        }
        LLMCommand::DeleteFile => match fs::remove_file(&change.filename) {
            Ok(_) => ChangeResult {
                filename: change.filename.to_string_lossy().into_owned(),
                success: true,
                message: "Deleted file".to_string(),
            },
            Err(e) => ChangeResult {
                filename: change.filename.to_string_lossy().into_owned(),
                success: false,
                message: format!("Failed to delete file: {}", e),
            },
        },
        _ => {
            let file_path = Path::new(&change.filename);
            match fs::read_to_string(file_path) {
                Ok(content) => match apply_change_to_content(&content, change) {
                    Ok(new_content) => match fs::write(file_path, new_content) {
                        Ok(_) => ChangeResult {
                            filename: change.filename.to_string_lossy().into_owned(),
                            success: true,
                            message: "Applied changes successfully".to_string(),
                        },
                        Err(e) => ChangeResult {
                            filename: change.filename.to_string_lossy().into_owned(),
                            success: false,
                            message: format!("Failed to write changes: {}", e),
                        },
                    },
                    Err(e) => ChangeResult {
                        filename: change.filename.to_string_lossy().into_owned(),
                        success: false,
                        message: format!("Failed to apply changes: {}", e),
                    },
                },
                Err(e) => ChangeResult {
                    filename: change.filename.to_string_lossy().into_owned(),
                    success: false,
                    message: format!("Failed to read file: {}", e),
                },
            }
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

async fn call_claude_bedrock(prompt: &str) -> String {
    /*
    let aws_config = aws_config::load_from_env().await;
    let bedrock_client = aws_sdk_bedrock::Client::new(&aws_config);

    let request = aws_sdk_bedrock::model::InvokeModelWithResponseStreamInput {
        body: prompt.as_bytes().to_vec(),
        model_id: "anthropic.claude-v2".to_string(),
        accept: "application/json".to_string(),
        content_type: "application/json".to_string(),
        ..Default::default()
    };

    let response = bedrock_client
        .invoke_model_with_response_stream()
        .model_id("anthropic.claude-v2")
        .body(request.body)
        .accept(request.accept)
        .content_type(request.content_type)
        .send()
        .await
        .expect("Failed to send request to Claude");

    let mut response_body = Vec::new();
    let mut stream = response.into_body();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.expect("Failed to read chunk from Claude response");
        response_body.extend_from_slice(&chunk);
    }

    let response_str =
        String::from_utf8(response_body).expect("Failed to convert response to UTF-8");
    response_str
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

fn make_context(
    maximum_context_tokens: usize,
    config: &Config,
) -> anyhow::Result<(String, Vec<ProcessedFile>)> {
    let mut content = String::new();
    let mut current_token_count = 0;
    let mut processed_files = Vec::new();
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
            let file_token_count = bpe.encode_with_special_tokens(&file_context).len();

            if current_token_count + file_token_count <= maximum_context_tokens {
                content.push_str(&file_context);
                current_token_count += file_token_count;
                processed_files.push(ProcessedFile {
                    name: path_str.to_string(),
                    token_count: file_token_count,
                });
            } else {
                break;
            }
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
        let command_token_count = bpe.encode_with_special_tokens(&command_context).len();

        if current_token_count + command_token_count <= maximum_context_tokens {
            content.push_str(&command_context);
            current_token_count += command_token_count;
            processed_files.push(ProcessedFile {
                name: format!("Command: {}", cmd),
                token_count: command_token_count,
            });
        } else {
            break;
        }
    }

    let content = content.replace('\n', "\r\n");

    println!(
        "Success loading context! Token count: {}",
        current_token_count
    );
    Ok((content, processed_files))
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
