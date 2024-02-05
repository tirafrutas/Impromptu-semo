import { AstNode, LangiumParser} from 'langium';
import { createImpromptuServices } from './language-server/impromptu-module.js';
import { Model, isModel } from './language-server/generated/ast.js';
import { NodeFileSystem } from 'langium/node';

// To retrieve the template files
import * as path from 'path';
import * as fs from 'fs';
import { ExtensionContext } from 'vscode';
import { generatePromptCode, AISystem, getPromptsList } from './cli/generate-prompt';

let Templates = new Map();

export interface Generator {
    // Load the Abstract Syntax Tree of the .prm active file
    generateCode(model : string | AstNode, aiSystem : string, prompt: string, context: ExtensionContext) : string | undefined;
    // Receives the parsed AST, generates the Python code, and returns it
    model2Code(model : Model, provider: string, template: string, prompt: string, context: ExtensionContext) : string | undefined;
}
 
/**
* Python code generator service main class
*/
export class CodeGenerator implements Generator {

    private readonly parser: LangiumParser;

    constructor(context: ExtensionContext) {       
        const services = createImpromptuServices(NodeFileSystem);
        this.parser = services.Impromptu.parser.LangiumParser;

        var fullFilePath = context.asAbsolutePath(path.join('resources', 'openai-chatgpt-template.py'));
        var template = fs.readFileSync(fullFilePath, "utf8");
        Templates.set(AISystem.ChatGPT, template);
        fullFilePath = context.asAbsolutePath(path.join('resources', 'stable-diffusion-template.py'));
        template = fs.readFileSync(fullFilePath, "utf8");
        Templates.set(AISystem.StableDiffusion, template);
    }

    getPromptsList(model: string) {
        const astNode = this.parser.parse(model).value;
        return (isModel(astNode) ? getPromptsList(astNode) : undefined);
    }

    generateCode(model : string, aiSystem: string, prompt: string) : string | undefined { // | AstNode) : string | undefined {
        //const astNode = (typeof(Model) == 'string' ? this.parser.parse(Model).value : Model);
        //return (isModel(astNode) ? this.model2Html(astNode) : undefined);
        const astNode = this.parser.parse(model).value;
        var template = Templates.get(aiSystem);
        return (isModel(astNode) ? this.model2Code(astNode, aiSystem, template, prompt) : undefined);
    }

    // Generation of the output code string
    model2Code(model : Model, aiSystem: string, template: string, prompt: string) : string | undefined {
        const promptCode = generatePromptCode(model, aiSystem, prompt);
        if (promptCode != null) {
            return template.replace('{PROMPT}', promptCode.filter(e => e !== '\n').filter(function(e){return e}).toString());
        }
        else
            return 'ERROR: Cannot generate prompt code.';
    }
}
 