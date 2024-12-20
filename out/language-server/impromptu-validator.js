"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.ImpromptuValidator = exports.registerValidationChecks = void 0;
const ast_1 = require("./generated/ast");
const fs_1 = __importDefault(require("fs"));
/**
 * Register custom validation checks.
 */
function registerValidationChecks(services) {
    const registry = services.validation.ValidationRegistry;
    const validator = services.validation.ImpromptuValidator;
    const checks = {
        Model: validator.checkModelWellFormedRules,
        Asset: validator.checkLanguageAsset,
        Parameters: validator.checkUniqueParams,
        AssetReuse: validator.checkAssetReuse,
        Multimodal: validator.checkMultimodalInputNotText,
        ImportedAsset: validator.checkImportedAsset,
        //CombinationTrait: validator.checkCombinationTrait,
        Language: validator.checkLanguage,
    };
    registry.register(checks, validator);
}
exports.registerValidationChecks = registerValidationChecks;
function check_loops_snippets(snippets, accept, og_asset) {
    if (snippets != undefined) {
        snippets.forEach(snippet => {
            if ((0, ast_1.isAssetReuse)(snippet.content)) {
                if (snippet.content.asset.ref)
                    if ((0, ast_1.isAsset)(snippet.content.asset.ref)) {
                        check_loops_asset(snippet.content.asset.ref, accept, og_asset);
                    }
                /*  It does not work: the imports are not fully upload yet
                else if (isAssetImport(snippet.content.asset.ref)){
                    check_loops_asset(get_imported_asset() , accept, og_asset);
                }*/
                if (snippet.content.pars && snippet.content.pars.pars)
                    check_loops_snippets(snippet.content.pars.pars, accept, og_asset);
            }
        });
    }
}
/**
 * Check whether there are infinite loops in the model due to the references or not. Recursive function
 * @param asset Asset where we are
 * @param og_asset Noted asset used to check if there is a loop
 */
function check_loops_asset(asset, accept, og_asset) {
    if (og_asset === null || og_asset === void 0 ? void 0 : og_asset.includes(asset)) {
        accept('error', "There is a recursive loop", { node: asset, property: 'name' });
    }
    else {
        let elements = [];
        if (!og_asset) {
            og_asset = [];
        }
        if ((0, ast_1.isPrompt)(asset)) {
            // Get all of snippets in a Prompt
            elements = asset.core.snippets;
            if (asset.prefix) {
                elements = elements.concat(asset.prefix.snippets);
            }
            if (asset.suffix) {
                elements = elements.concat(asset.suffix.snippets);
            }
        }
        else if ((0, ast_1.isComposer)(asset)) {
            // Get all of snippets in a Composer
            elements = asset.contents.snippets;
        }
        else if ((0, ast_1.isImportedAsset)(asset)) {
            // Get the asset ImportedAsset references
            //check_loops_asset(get_imported_asset(asset), og_asset);
        }
        if (elements) {
            og_asset.push(asset);
            check_loops_snippets(elements, accept, og_asset);
        }
    }
}
/**
 * Implementation of custom validations.
 */
class ImpromptuValidator {
    /* checkPersonStartsWithCapital(person: Person, accept: ValidationAcceptor): void {
        if (person.name) {
            const firstChar = person.name.substring(0, 1);
            if (firstChar.toUpperCase() !== firstChar) {
                accept('warning', 'Person name should start with a capital.', { node: person, property: 'name' });
            }
        }
    } */
    checkMultimodalInputNotText(input, accept) {
        if (input.format == 'text') {
            accept('error', `Textual inputs should be defined as parameters, not multi-modal inputs. \nUse '@par' instead of '$par:text'.`, { node: input, property: 'format' });
        }
    }
    /**
     * Set of rules the Model should fullfil
     * @param model
     * @param accept
     */
    checkModelWellFormedRules(model, accept) {
        this.checkUniqueAssets(model, accept);
        this.checkByExpressionValidators(model, accept);
        this.checkNoCyclesInVersions(model, accept);
        this.checkNoCyclesInRefines(model, accept);
        this.checkNoRecursivity(model, accept);
    }
    checkUniqueAssets(model, accept) {
        // create a set of visited assets
        // and report an error when we see one we've already seen
        const reported = new Set();
        model.assets.forEach(a => {
            if (reported.has(a.name)) {
                accept('error', `Asset has non-unique name '${a.name}'.`, { node: a, property: 'name' });
            }
            reported.add(a.name);
        });
        // It also has to consider the imported assets
        const reported_imports = new Set();
        model.imports.forEach(import_line => {
            import_line.asset_name.forEach(a => {
                if (reported.has(a.name)) {
                    accept('error', `Asset has non-unique name '${a.name}'.`, { node: a, property: 'name' });
                }
                else if (reported_imports.has(a.name)) {
                    accept('error', `Two imported assets has non-unique name '${a.name}'.`, { node: a, property: 'name' });
                }
                reported_imports.add(a.name);
            });
        });
    }
    checkNoCyclesInVersions(model, accept) {
        model.assets.forEach(a => {
            if (a.priorVersion != undefined) {
                let node = model.assets.filter(p => { var _a; return p.name == ((_a = a.priorVersion) === null || _a === void 0 ? void 0 : _a.$refText); })[0];
                while (node != undefined) {
                    if (node.name == a.name) {
                        accept('error', `Cannot be cycles in prior version relationship.`, { node: a, property: 'priorVersion' });
                        break;
                    }
                    if (node.priorVersion != undefined)
                        node = model.assets.filter(a => { var _a; return a.name == ((_a = node.priorVersion) === null || _a === void 0 ? void 0 : _a.$refText); })[0];
                    else
                        break;
                }
            }
        });
    }
    checkNoRecursivity(model, accept) {
        model.assets.forEach(asset => {
            check_loops_asset(asset, accept);
        });
    }
    checkNoCyclesInRefines(model, accept) {
        model.assets.forEach(a => {
            if (a.refines != undefined) {
                let node = model.assets.filter(p => { var _a; return p.name == ((_a = a.refines) === null || _a === void 0 ? void 0 : _a.$refText); })[0];
                while (node != undefined) {
                    if (node.name == a.name) {
                        accept('error', `Cannot be cycles in refinement relationship.`, { node: a, property: 'refines' });
                        break;
                    }
                    if (node.refines != undefined)
                        node = model.assets.filter(a => { var _a; return a.name == ((_a = node.refines) === null || _a === void 0 ? void 0 : _a.$refText); })[0];
                    else
                        break;
                }
            }
        });
    }
    checkByExpressionValidators(model, accept) {
        model.assets.forEach(a => {
            if ((0, ast_1.isByExpressionOutputTesting)(a)) {
                const validator = model.assets.filter(p => (0, ast_1.isPrompt)(p) && p.name == a.validator.$refText)[0];
                // verify that the output media is text
                if (validator && validator.output != 'text')
                    accept('error', `The output media of validator must be of type text.`, { node: validator, property: 'output' });
                // verify that a validator does not have a validator
                if (validator && (0, ast_1.isByExpressionOutputTesting)(validator))
                    accept('error', `A validator cannot have an output validation itself.`, { node: validator, property: 'validator' });
            }
        });
    }
    /**
     * Validations done to the `AssetReuse` objects:
     *
     * - The number of parameters in `pars` match with the number of parameters defined in the referenced `Asset`
     * referenced
     *
     * @param assetReuse
     * @param accept
     */
    checkAssetReuse(assetReuse, accept) {
        const values = assetReuse.pars;
        const ogAsset = assetReuse.asset.ref;
        if (!((0, ast_1.isChain)(ogAsset) || (0, ast_1.isAssetImport)(ogAsset))) {
            if ((ogAsset === null || ogAsset === void 0 ? void 0 : ogAsset.pars.pars.length) != (values === null || values === void 0 ? void 0 : values.pars.length)) {
                accept('error', `The number of variables (${values === null || values === void 0 ? void 0 : values.pars.length}) does not match with the number of variables of the referenced Asset (${ogAsset === null || ogAsset === void 0 ? void 0 : ogAsset.pars.pars.length})`, { node: assetReuse });
            }
        }
        else if ((0, ast_1.isChain)(ogAsset)) {
            if (values) {
                accept('error', `The Asset referenced is a Chain. A Chain does not have nay parameters.`, { node: assetReuse });
            }
        }
    }
    checkUniqueParams(parset, accept) {
        // create a set of visited parameters
        // and report an error when we see one we've already seen
        const reported = new Set();
        parset.pars.forEach(p => {
            if (reported.has(p.name)) {
                accept('error', `Input has non-unique name '${p.name}'.`, { node: p, property: 'name' });
            }
            reported.add(p.name);
        });
    }
    /**
     * Validations for an ImportedAsset:
     * - The file it refecences (`imported_asset.library`) exists.
     * - The prompt it tries to import (`imported_asset.name`) exists in the told file.
     * @param imported_asset
     * @param accept
     */
    checkImportedAsset(imported_asset, accept) {
        // I- The file it refecences (`imported_asset.library`) exists.
        const library = imported_asset.library.split(".").join("/"); // Convert the Qualified name into a relative path
        let workspace_path = process.env.WORKSPACE;
        if (!workspace_path) {
            workspace_path = process.cwd();
        }
        let uri_array = workspace_path.split("/"); //("\\")
        let last = 'build_files';
        uri_array.push(last);
        workspace_path = uri_array.join("/"); //("\\");
        //const uri= uri_array?.join("/")
        if (fs_1.default.existsSync(workspace_path + '/' + library)) {
            accept('error', `The file ` + library + ` exists, but the file format ".prm" has to be erased.`, { node: imported_asset });
        }
        else if (!fs_1.default.existsSync(workspace_path + '/' + library + '.prm')) {
            accept('error', `The library ` + workspace_path + library + ` does not exist.`, { node: imported_asset });
        }
        else {
            // II- The prompt it tries to import (`imported_asset.name`) exists in the told file.
            let buffer = fs_1.default.readFileSync(workspace_path + '/' + library + '.prm');
            imported_asset.asset_name.forEach(asset_import => {
                let assetRegex = new RegExp(`.*\\s${asset_import.name}\\s*\\(.*`);
                if (!assetRegex.test(buffer.toString())) {
                    accept('error', `The prompt ` + asset_import.name + ` is not included in ` + library, { node: imported_asset });
                }
            });
        }
    }
    //     isDescendant(model: Model, asset: Asset, accept: ValidationAcceptor) {
    //         var node = model.assets.filter(a => a.name == asset.priorVersion.$refText)[0];
    //         accept('error', `Asset name == '${asset.name}'; prior version name == '${node.name}'.`,  {node: asset, property: 'priorVersion'});
    //         while (node != null) {
    //             if (node.name == asset.name) {
    //                 return true;
    //             }
    //             node = model.assets.filter(a => a.name == node.priorVersion.$refText)[0];
    //         }
    //         return false;
    //    }
    /**
     * Checks that the CombinationTrait `snippet` has more than one parameter
     * @param snippet
     * @param accept
     */
    checkCombinationTrait(snippet, accept) {
        const n_parameters = snippet.contents.length;
        if (n_parameters < 2) {
            accept('error', 'A combination trait needs at least two inputs', { node: snippet });
        }
    }
    ;
    /**
     * Checks that the langugae selected is supported
     * @param language
     * @param accept
     */
    checkLanguage(language, accept) {
        if (!findLanguage(language.name)) {
            accept('error', `Language is not supported.`, { node: language });
        }
    }
    checkLanguageAsset(asset, accept) {
        if (!(0, ast_1.isChain)(asset)) {
            if (asset.language) { // If declares the language individually
                if (!findLanguage(asset.language)) {
                    accept('error', `Language is not supported.`, { node: asset }); // Maybe change AssetLanguage to be an asset
                }
                else {
                    if (asset.language == asset.$container.language.name) {
                        accept('hint', `Langugae redundant. The file's language is already ${asset.language}`, { node: asset }); // Maybe change AssetLanguage to be an asset
                    }
                }
            }
        }
    }
}
exports.ImpromptuValidator = ImpromptuValidator;
/**
 * Checks if the given language is in the file `lang.json`
 * @param language_name
 * @returns
 */
function findLanguage(language_name) {
    let workspace_path = process.env.WORKSPACE;
    if (!workspace_path) {
        workspace_path = process.cwd();
    }
    let found = false;
    const json_file = fs_1.default.readFileSync(workspace_path + '/languages/lang.json');
    const json = JSON.parse(json_file.toString());
    json.forEach((element) => {
        if (element["language"] == language_name || element["code"] == language_name) {
            found = true;
        }
    });
    return found;
}
//# sourceMappingURL=impromptu-validator.js.map