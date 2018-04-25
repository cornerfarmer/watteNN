jQuery(document).ready(function () {
    initialize();
});

var cards = [];
var cards_left = [];
var players = [{hand_cards: [], board_card: null, tricks: 0}, {hand_cards: [], board_card: null, tricks: 0}];
var number_of_hand_cards = 5;
var current_player = 0;
var next_start_player = 1;
var last_tricks = [];
var colors = {
    EICHEL: 0,
    GRUEN: 1,
    HERZ: 2,
    SCHELLN: 3
};
var minimal = false;

var values = {
    SAU: 7,
    KOENIG: 6,
    OBER: 5,
    UNTER: 4,
    ZEHN: 3,
    NEUN: 2,
    ACHT: 1,
    SIEBEN: 0
};
var play_model, choose_model;

var action_types = {
    CHOOSE_VALUE: 0,
    CHOOSE_COLOR: 1,
    DRAW_CARD: 2
};
var next_action_type;
var chosen_value, chosen_color;

function initialize() {
    play_model = new KerasJS.Model({
      filepath: 'play-twoNetsv8.bin'
    });

    choose_model = new KerasJS.Model({
      filepath: 'choose-twoNetsv8.bin'
    });

    var index = 0;
    for (var color = 0; color < (minimal ? 2 : 4); color++) {
        for (var value = 7; value >= (minimal ? 4 : 0); value--) {
            //cards.push({color: colors[color], value: values[value]})
            cards.push({color: color, value: value, index: index++});
        }
    }

    reset();
}

function shuffle(a) {
    var j, x, i;
    for (i = a.length; i; i--) {
        j = Math.floor(Math.random() * i);
        x = a[i - 1];
        a[i - 1] = a[j];
        a[j] = x;
    }
}

function reset() {
    cards_left = cards.slice(0);
    shuffle(cards_left);
    for (var i = 0; i < players.length; i++) {
        players[i].hand_cards = [];
        players[i].board_card = null;
        players[i].tricks = 0;
        for (var c = 0; c < number_of_hand_cards; c++)
            players[i].hand_cards.push(cards_left.pop());
    }
    current_player = next_start_player;
    next_start_player = 1 - next_start_player;
    last_tricks = [];
    next_action_type = action_types.CHOOSE_VALUE;
    refreshState("");
    refreshView();


    if (current_player === 0)
        setTimeout(runAI, 1000);
}

function createCardHTML(card, number_in_hand_cards) {
    return '<div class="card" data-card-nr="' + number_in_hand_cards + '" style="background-image: url(\'../cards/' + filenameFromCard(card) + '.png\')"></div>'
}

function filenameFromCard(card) {
    var filename = "";
    if (card.color === colors.EICHEL)
        filename += "E";
    else if (card.color === colors.GRUEN)
        filename += "G";
    else if (card.color === colors.HERZ)
        filename += "H";
    else if (card.color === colors.SCHELLN)
        filename += "S";

    if (card.value === values.SAU)
        filename += "A";
    else if (card.value === values.KOENIG)
        filename += "K";
    else if (card.value === values.OBER)
        filename += "O";
    else if (card.value === values.UNTER)
        filename += "U";
    else if (card.value === values.ZEHN)
        filename += "10";
    else if (card.value === values.NEUN)
        filename += "9";
    else if (card.value === values.ACHT)
        filename += "8";
    else if (card.value === values.SIEBEN)
        filename += "7";
    return filename
}

function color_to_string(color) {
    if (color === colors.EICHEL)
        return "EICHEL";
    else if (color === colors.GRUEN)
        return "GRUEN";
    else if (color === colors.HERZ)
        return "HERZ";
    else if (color === colors.SCHELLN)
        return "SCHELLN";
}

function value_to_string(value) {
    if (value === values.SAU)
        return "SAU";
    else if (value === values.KOENIG)
        return "KOENIG";
    else if (value === values.OBER)
        return "OBER";
    else if (value === values.UNTER)
        return "UNTER";
    else if (value === values.ZEHN)
        return "ZEHN";
    else if (value === values.NEUN)
        return "NEUN";
    else if (value === values.ACHT)
        return "ACHT";
    else if (value === values.SIEBEN)
        return "SIEBEN";
}

function set_card(card) {
    if (next_action_type === action_types.CHOOSE_VALUE) {
        chosen_value = card.value;
        next_action_type = action_types.CHOOSE_COLOR;
        current_player = 1 - current_player;
        action = "Chosen value is " + value_to_string(chosen_value);

        if (current_player === 0)
            setTimeout(runAI, 1000);
    } else if (next_action_type === action_types.CHOOSE_COLOR) {
        chosen_color = card.color;
        next_action_type = action_types.DRAW_CARD;
        current_player = 1 - current_player;
        action = "Chosen value/color is " + value_to_string(chosen_value) + " " + color_to_string(chosen_color);

        if (current_player === 0)
            setTimeout(runAI, 1000);
    } else {
        var index = players[current_player].hand_cards.indexOf(card);
        var action = "";
        if (index >= 0) {
            if (players[0].board_card !== null && players[1].board_card !== null) {
                players[0].board_card = null;
                players[1].board_card = null;
            }

            players[current_player].board_card = players[current_player].hand_cards.splice(index, 1)[0];

            if (players[0].board_card !== null && players[1].board_card !== null) {
                var best_player = match(players[1 - current_player].board_card, players[current_player].board_card);

                last_tricks.push(players[0].board_card);
                last_tricks.push(players[1].board_card);

                if (best_player === 0)
                    current_player = 1 - current_player;

                players[self.current_player].tricks += 1
            }
            else
                current_player = 1 - current_player;

            if (players[0].tricks === 3)
            {
                action = "AI won!";
                setTimeout(reset, 1000);
            }
            else if (players[1].tricks === 3)
            {
                action = "Human won!";
                setTimeout(reset, 1000);
            }
            else if (players[1].hand_cards.length + players[0].hand_cards.length === 0)
                setTimeout(reset, 1000);
            else if (current_player === 0)
                setTimeout(runAI, 1000);
        }
        else
        {
            action = "AI made an invalid move.";
            setTimeout(reset, 1000);
        }
    }
    refreshState(action);
    refreshView();
}


function refreshView() {
    $("#player-ai .cards").empty();
    for (var i = 0; i < players[0].hand_cards.length; i++) {
        $("#player-ai .cards").append(createCardHTML(players[0].hand_cards[i], i));
    }
    $("#boardcard-ai").empty();
    if (players[0].board_card !== null)
        $("#boardcard-ai").append(createCardHTML(players[0].board_card, i));

    $("#player-human .cards").empty();
    for (var i = 0; i < players[1].hand_cards.length; i++) {
        $("#player-human .cards").append(createCardHTML(players[1].hand_cards[i], i));
    }
    $("#boardcard-human").empty();
    if (players[1].board_card !== null)
        $("#boardcard-human").append(createCardHTML(players[1].board_card, i));

    $("#player-human .cards .card").click(function () {
        if (current_player === 1) {
            set_card(players[1].hand_cards[$(this).attr('data-card-nr')]);
        }
    })

}

function addTrickArray(arr, player)
{
    if (player.tricks === 0) {
        arr.push(0);
        arr.push(0);
    }
    else if (player.tricks === 1) {
        arr.push(1);
        arr.push(0);
    }
    else if (player.tricks === 2) {
        arr.push(0);
        arr.push(1);
    }
    else if (player.tricks === 3) {
        arr.push(1);
        arr.push(1);
    }
}

function indexOfMax(arr)
{
    var max = arr[0];
    var maxIndex = 0;
    for (var i = 0; i < arr.length; i++)
    {
        if (arr[i] > max)
        {
            max = arr[i];
            maxIndex = i;
        }
    }
    return maxIndex;
}


function runAI()
{
    var number_of_sets;
    if (next_action_type === action_types.DRAW_CARD)
        number_of_sets = 2 + 8 + 2;
    else
        number_of_sets = 2;

    obs_cube = [];
    for (var c = 0; c < 4 * 8 * number_of_sets; c++) {
        obs_cube.push(0);
    }
    for (var i = 0; i < players[0].hand_cards.length; i++) {
        obs_cube[players[0].hand_cards[i].color * 8 * number_of_sets + players[0].hand_cards[i].value * number_of_sets + 0] = 1;
    }

    if (next_action_type === action_types.DRAW_CARD) {
        if (players[1].board_card !== null)
            obs_cube[players[1].board_card.color * 8 * number_of_sets + players[1].board_card.value * number_of_sets + 1] = 1;

        for (var i = Math.max(0, last_tricks.length - 8); i < last_tricks.length; i++)
            obs_cube[last_tricks[i].color * 8 * number_of_sets + last_tricks[i].value * number_of_sets + (2 + ((last_tricks.length - 1) / 2 - i / 2) * 2 + (current_player === 1 ? (1 - i % 2) : (i % 2)))] = 1

        for (var color = 0; color < 4; color++)
            obs_cube[color * 8 * number_of_sets + chosen_value * number_of_sets + number_of_sets - 2] = 1;

        for (var value = 7; value >= 0; value--)
            obs_cube[chosen_color * 8 * number_of_sets + value * number_of_sets + number_of_sets - 1] = 1;
    } else {
        if (next_action_type === action_types.CHOOSE_COLOR) {
            for (var color = 0; color < 4; color++)
                obs_cube[color * 8 * number_of_sets + chosen_value * number_of_sets + 1] = 1;
        }
    }

    console.log(obs_cube);
    obs_flat = [];
    addTrickArray(obs_flat, players[0]);
    addTrickArray(obs_flat, players[1]);

    var model = next_action_type === action_types.DRAW_CARD ? play_model : choose_model;
    model.ready()
    .then(() => {
        const inputData = {};

        if (next_action_type === action_types.DRAW_CARD) {
            inputData["input_5"] = new Float32Array(obs_cube);
            inputData["input_6"] = new Float32Array(obs_flat);
        } else {
            inputData["input_4"] = new Float32Array(obs_cube);
        }


        // make predictions
        return model.predict(inputData)
    }).then(outputData => {
        console.log(outputData);
        var output = outputData[next_action_type === action_types.DRAW_CARD ? 'dense_14' : 'dense_10'];
        var max_i = 0;
        for (var i = 1; i < players[current_player].hand_cards.length; i++) {
            if (output[players[current_player].hand_cards[i].index] > output[players[current_player].hand_cards[max_i].index])
                max_i = i;
        }
        set_card(players[current_player].hand_cards[max_i]);
    })
}

function match(first_card, second_card)
{
    if (getCardValue(first_card, first_card) >= getCardValue(second_card, first_card))
        return 0;
    else
        return 1;
}

function getCardValue(card, first_card)
{
    if (card.color === colors.HERZ && card.value === values.KOENIG)
        return 20;
    else if (card.color === colors.SCHELLN && card.value === values.SIEBEN)
        return 19;
    else if (card.color === colors.EICHEL && card.value === values.SIEBEN)
        return 18;

    if (card.color === chosen_color  && card.value === chosen_value)
        return 17;
    else if (card.value === chosen_value)
        return 16;
    else if (card.color === chosen_color)
        return card.value + 9;
    else if (card.color === first_card.color)
        return card.value + 1;
    else
        return 0;
}

function refreshState(lastAction)
{
    var action_to_do;
    if (next_action_type === action_types.CHOOSE_VALUE)
        action_to_do = "to choose the value";
    else if (next_action_type === action_types.CHOOSE_COLOR)
        action_to_do = "to choose the color";
    else
        action_to_do = "to draw a card (" + value_to_string(chosen_value) + " " + color_to_string(chosen_color) + ")";
    $("#state").html((current_player === 1 ? "It's your turn" : "It's AI turn") + " " + action_to_do + (lastAction !== "" ? " - " + lastAction : ""));
}